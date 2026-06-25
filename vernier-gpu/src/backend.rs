//! [`GpuBackend`]: the `rustfft`/`ndarray` implementation of
//! [`ComputeBackend`](vernier_core::ComputeBackend).

use std::cell::RefCell;
use std::sync::Arc;

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, Real, Result, VernierError};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{VulkanLibrary, sync};

use crate::buffer::GpuBuffer;

/// GPU optimized backend.
pub struct GpuBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    builder: RefCell<Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>>,
    ccx: ComputeContext,
    instance: Arc<Instance>,
    memory_allocator: Arc<StandardMemoryAllocator>,
}

struct ComputeContext {
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    fft_pipeline: Arc<ComputePipeline>,
    ifft_pipeline: Arc<ComputePipeline>,
    blur_pipeline: Arc<ComputePipeline>,
    extract_phase_pipeline: Arc<ComputePipeline>,
    filter_pipeline: Arc<ComputePipeline>,
    peak_search_pipeline: Arc<ComputePipeline>,
}

impl GpuBackend {
    /// Creates a new CPU backend.
    pub fn new() -> Self {
        let library = VulkanLibrary::new().unwrap();

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let fft_pipeline = {
            let cs = fft_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let ifft_pipeline = {
            let cs = ifft_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let blur_pipeline = {
            let cs = blur_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let extract_phase_pipeline = {
            let cs = extract_phase_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let filter_pipeline = {
            let cs = filter_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let peak_search_pipeline = {
            let cs = peak_search_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        // let layout = &filter_pipeline.layout().set_layouts()[0];
        // let filter_descriptor_set = DescriptorSet::new(
        //     descriptor_set_allocator.clone(),
        //     layout.clone(),
        //     [WriteDescriptorSet::buffer(0, spectrum.buffer.clone())],
        //     [],
        // )
        // .unwrap();

        // let layout = &peak_search_pipeline.layout().set_layouts()[0];
        // let peak_search_descriptor_set = DescriptorSet::new(
        //     descriptor_set_allocator.clone(),
        //     layout.clone(),
        //     [WriteDescriptorSet::buffer(0, spectrum.buffer.clone())],
        //     [],
        // )
        // .unwrap();

        Self {
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            builder: RefCell::new(None),
            ccx: ComputeContext {
                descriptor_set_allocator,
                fft_pipeline,
                ifft_pipeline,
                blur_pipeline,
                extract_phase_pipeline,
                filter_pipeline,
                peak_search_pipeline,
            },
        }
    }
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for GpuBackend {
    type Buffer2D = GpuBuffer;

    fn init(&mut self) -> Result<()> {
        match AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ) {
            Ok(builder) => {
                self.builder = RefCell::new(Some(builder));
                Ok(())
            }
            Err(err) => Err(VernierError::Backend(err.to_string())),
        }
    }

    fn exec(&mut self) -> Result<()> {
        let command_buffer = {
            let builder = self
                .builder
                .borrow_mut()
                .take()
                .ok_or_else(|| VernierError::Backend("Backend not initialized".to_string()))?;
            builder.build().unwrap()
        };

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();
        Ok(())
    }

    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<Self::Buffer2D> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            if !layout.is_contiguous() {
                return Err(VernierError::NonContiguous {
                    stride: layout.row_stride,
                    width: layout.width,
                });
            }

            let data = Vec::from(data);
            let host_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST,
                    ..Default::default()
                },
                data,
            )
            .unwrap();

            let num_elements = layout.width * layout.height;
            let buffer_size = (num_elements as u64).next_power_of_two();
            let device_buffer = Buffer::new_slice::<Complex32>(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                buffer_size,
            )
            .unwrap();

            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    host_buffer.clone(),
                    device_buffer.clone(),
                ))
                .unwrap();

            Ok(GpuBuffer {
                buffer: device_buffer,
                width: layout.width,
                height: layout.height,
            })
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    fn download(&self, buffer: &Self::Buffer2D) -> Result<Vec<Complex32>> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            let host_buffer = Buffer::new_slice::<Complex32>(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                buffer.size() as u64,
            )
            .unwrap();

            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    buffer.buffer.clone(),
                    host_buffer.clone(),
                ))
                .unwrap();
            Ok(host_buffer.read().unwrap().to_vec())
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    fn fft2d(&self, buffer: &mut Self::Buffer2D) -> Result<()> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            let layout = self.ccx.fft_pipeline.layout().set_layouts()[0].clone();
            let fft_descriptor_set = DescriptorSet::new(
                self.ccx.descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(0, buffer.buffer.clone())],
                [],
            )
            .unwrap();

            let push_constants = fft_shader::PushConstantData {
                width: buffer.width as u32,
                height: buffer.height as u32,
            };

            builder
                .bind_pipeline_compute(self.ccx.fft_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.ccx.fft_pipeline.layout().clone(),
                    0,
                    fft_descriptor_set.clone(),
                )
                .unwrap()
                .push_constants(self.ccx.fft_pipeline.layout().clone(), 0, push_constants)
                .unwrap();

            Ok(())
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    fn ifft2d(&self, buffer: &mut Self::Buffer2D) -> Result<()> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            let layout = self.ccx.ifft_pipeline.layout().set_layouts()[0].clone();
            let ifft_descriptor_set = DescriptorSet::new(
                self.ccx.descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(0, buffer.buffer.clone())],
                [],
            )
            .unwrap();

            let push_constants = ifft_shader::PushConstantData {
                width: buffer.width as u32,
                height: buffer.height as u32,
            };

            builder
                .bind_pipeline_compute(self.ccx.ifft_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.ccx.ifft_pipeline.layout().clone(),
                    0,
                    ifft_descriptor_set.clone(),
                )
                .unwrap()
                .push_constants(self.ccx.ifft_pipeline.layout().clone(), 0, push_constants)
                .unwrap();

            Ok(())
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    fn extract_phase(&self, buffer: &Self::Buffer2D) -> Result<Self::Buffer2D> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            let num_elements = buffer.size();
            let buffer_size = (num_elements as u64).next_power_of_two();
            let device_buffer = Buffer::new_slice::<Complex32>(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                buffer_size,
            )
            .unwrap();

            let layout = self.ccx.extract_phase_pipeline.layout().set_layouts()[0].clone();
            let ifft_descriptor_set = DescriptorSet::new(
                self.ccx.descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, buffer.buffer.clone()),
                    WriteDescriptorSet::buffer(1, device_buffer.clone()),
                ],
                [],
            )
            .unwrap();

            let push_constants = ifft_shader::PushConstantData {
                width: buffer.width as u32,
                height: buffer.height as u32,
            };

            builder
                .bind_pipeline_compute(self.ccx.extract_phase_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.ccx.extract_phase_pipeline.layout().clone(),
                    0,
                    ifft_descriptor_set.clone(),
                )
                .unwrap()
                .push_constants(
                    self.ccx.extract_phase_pipeline.layout().clone(),
                    0,
                    push_constants,
                )
                .unwrap();

            Ok(GpuBuffer {
                buffer: device_buffer,
                width: buffer.width,
                height: buffer.height,
            })
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    fn filter(
        &self,
        buffer: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            let layout = self.ccx.ifft_pipeline.layout().set_layouts()[0].clone();
            let ifft_descriptor_set = DescriptorSet::new(
                self.ccx.descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(0, buffer.buffer.clone())],
                [],
            )
            .unwrap();

            let push_constants = ifft_shader::PushConstantData {
                width: buffer.width as u32,
                height: buffer.height as u32,
            };

            builder
                .bind_pipeline_compute(self.ccx.ifft_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.ccx.ifft_pipeline.layout().clone(),
                    0,
                    ifft_descriptor_set.clone(),
                )
                .unwrap()
                .push_constants(self.ccx.ifft_pipeline.layout().clone(), 0, push_constants)
                .unwrap();

            Ok(())
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    fn peak_search(
        &self,
        buffer: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<Self::Buffer2D>> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            let num_elements = buffer.size();
            let buffer_size = (num_elements as u64).next_power_of_two();
            let device_buffer = Buffer::new_slice::<Complex32>(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                buffer_size,
            )
            .unwrap();

            let layout = self.ccx.peak_search_pipeline.layout().set_layouts()[0].clone();
            let peak_search_descriptor_set = DescriptorSet::new(
                self.ccx.descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, buffer.buffer.clone()),
                    WriteDescriptorSet::buffer(1, device_buffer.clone()),
                ],
                [],
            )
            .unwrap();

            let push_constants = ifft_shader::PushConstantData {
                width: buffer.width as u32,
                height: buffer.height as u32,
            };

            builder
                .bind_pipeline_compute(self.ccx.peak_search_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.ccx.peak_search_pipeline.layout().clone(),
                    0,
                    peak_search_descriptor_set.clone(),
                )
                .unwrap()
                .push_constants(
                    self.ccx.peak_search_pipeline.layout().clone(),
                    0,
                    push_constants,
                )
                .unwrap();

            todo!()
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    /// Separable 2D Gaussian blur on a real-valued array, in place.
    fn gaussian_blur_2d(&self, buffer: &mut Self::Buffer2D, sigma: Real) -> Result<()> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            let layout = self.ccx.blur_pipeline.layout().set_layouts()[0].clone();
            let ifft_descriptor_set = DescriptorSet::new(
                self.ccx.descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(0, buffer.buffer.clone())],
                [],
            )
            .unwrap();

            let push_constants = blur_shader::PushConstantData {
                width: buffer.width as u32,
                height: buffer.height as u32,
            };

            builder
                .bind_pipeline_compute(self.ccx.blur_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.ccx.ifft_pipeline.layout().clone(),
                    0,
                    ifft_descriptor_set.clone(),
                )
                .unwrap()
                .push_constants(self.ccx.blur_pipeline.layout().clone(), 0, push_constants)
                .unwrap();

            Ok(())
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    fn bandpass_filter(&self, buffer: &mut Self::Buffer2D, sigma: Real) -> Result<()> {
        if let Some(ref mut builder) = *self.builder.borrow_mut() {
            todo!()
        } else {
            Err(VernierError::Backend("Backend not initialized".to_string()))
        }
    }

    fn name(&self) -> &str {
        "gpu-vulkan-fft"
    }
}

mod fft_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/fft.glsl",
        include: ["."],
        spirv_version: "1.3"
    }
}

mod ifft_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/ifft.glsl",
        include: ["."],
        spirv_version: "1.3"
    }
}

mod blur_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/blur.glsl",
        include: ["."],
        spirv_version: "1.3"
    }
}

mod extract_phase_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/extract_phase.glsl",
        include: ["."],
        spirv_version: "1.3"
    }
}

mod filter_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/filter.glsl",
        include: ["."],
        spirv_version: "1.3"
    }
}

mod peak_search_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/peak_search.glsl",
        include: ["."],
        spirv_version: "1.3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use vernier_core::buffer::BufferLayout;

    fn checkerboard(w: usize, h: usize) -> (Vec<Complex32>, BufferLayout) {
        let layout = BufferLayout::packed(w, h);
        let mut v = Vec::with_capacity(layout.len());
        for r in 0..h {
            for c in 0..w {
                let val = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
                v.push(Complex32::new(val, 0.0));
            }
        }
        (v, layout)
    }

    #[test]
    fn fft_then_ifft_is_identity() {
        let backend = GpuBackend::new();
        let (data, layout) = checkerboard(8, 8);
        let mut buf = backend.upload(&data, layout).unwrap();

        backend.fft2d(&mut buf).unwrap();
        backend.ifft2d(&mut buf).unwrap();

        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-4);
        }
    }

    // #[test]
    // fn argmax_finds_the_dc_spike() {
    //     // A constant image has all energy at the DC bin (index 0) after FFT.
    //     let backend = GpuBackend::new();
    //     let layout = BufferLayout::packed(8, 8);
    //     let data = vec![Complex32::new(1.0, 0.0); layout.len()];
    //     let mut buf = backend.upload(&data, layout).unwrap();

    //     backend.fft2d(&mut buf).unwrap();
    //     let (idx, _mag) = backend.argmax_magnitude(&buf).unwrap();
    //     assert_eq!(idx, 0, "all energy should sit at the DC bin");
    // }

    #[test]
    fn non_square_round_trips() {
        // Guards the row/column length bookkeeping (w != h).
        let backend = GpuBackend::new();
        let (data, layout) = checkerboard(16, 4);
        let mut buf = backend.upload(&data, layout).unwrap();
        backend.fft2d(&mut buf).unwrap();
        backend.ifft2d(&mut buf).unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
        }
    }
}
