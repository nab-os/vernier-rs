//! [`GpuBackend`] and [`GpuJob`]: the Vulkano/GLSL implementations of
//! [`ComputeBackend`](vernier_core::ComputeBackend) and [`ComputeJob`](vernier_core::ComputeJob).

use std::sync::Arc;

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, ComputeJob, Real, Result, VernierError};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
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

// ---------------------------------------------------------------------------
// Shared compute pipelines
// ---------------------------------------------------------------------------

struct ComputeContext {
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    fft_pipeline: Arc<ComputePipeline>,
    ifft_pipeline: Arc<ComputePipeline>,
    blur_pipeline: Arc<ComputePipeline>,
    extract_phase_pipeline: Arc<ComputePipeline>,
    filter_pipeline: Arc<ComputePipeline>,
    bandpass_pipeline: Arc<ComputePipeline>,
    magnitude_pipeline: Arc<ComputePipeline>,
    argmax_local_pipeline: Arc<ComputePipeline>,
    argmax_global_pipeline: Arc<ComputePipeline>,
    band_angular_filter_pipeline: Arc<ComputePipeline>,
    order_peaks_pipeline: Arc<ComputePipeline>,
}

fn build_pipeline(
    device: Arc<Device>,
    shader_module: Arc<vulkano::shader::ShaderModule>,
) -> Arc<ComputePipeline> {
    let cs = shader_module.entry_point("main").unwrap();
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
}

// ---------------------------------------------------------------------------
// GpuBackend
// ---------------------------------------------------------------------------

/// GPU compute backend (Vulkano).
pub struct GpuBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    ccx: ComputeContext,
}

impl GpuBackend {
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

        let fft_pipeline =
            build_pipeline(device.clone(), fft_shader::load(device.clone()).unwrap());
        let ifft_pipeline =
            build_pipeline(device.clone(), ifft_shader::load(device.clone()).unwrap());
        let blur_pipeline =
            build_pipeline(device.clone(), blur_shader::load(device.clone()).unwrap());
        let extract_phase_pipeline =
            build_pipeline(device.clone(), extract_phase_shader::load(device.clone()).unwrap());
        let filter_pipeline =
            build_pipeline(device.clone(), filter_shader::load(device.clone()).unwrap());
        let bandpass_pipeline =
            build_pipeline(device.clone(), bandpass_shader::load(device.clone()).unwrap());
        let magnitude_pipeline =
            build_pipeline(device.clone(), magnitude_shader::load(device.clone()).unwrap());
        let argmax_local_pipeline =
            build_pipeline(device.clone(), argmax_local_shader::load(device.clone()).unwrap());
        let argmax_global_pipeline =
            build_pipeline(device.clone(), argmax_global_shader::load(device.clone()).unwrap());
        let band_angular_filter_pipeline = build_pipeline(
            device.clone(),
            band_angular_filter_shader::load(device.clone()).unwrap(),
        );
        let order_peaks_pipeline =
            build_pipeline(device.clone(), peak_search_shader::load(device.clone()).unwrap());

        Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            ccx: ComputeContext {
                descriptor_set_allocator,
                fft_pipeline,
                ifft_pipeline,
                blur_pipeline,
                extract_phase_pipeline,
                filter_pipeline,
                bandpass_pipeline,
                magnitude_pipeline,
                argmax_local_pipeline,
                argmax_global_pipeline,
                band_angular_filter_pipeline,
                order_peaks_pipeline,
            },
        }
    }

    /// Allocates a DEVICE-local storage buffer of `n` complex elements.
    fn alloc_device_buffer(&self, n: usize) -> Subbuffer<[Complex32]> {
        Buffer::new_slice::<Complex32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            n as u64,
        )
        .unwrap()
    }

    fn make_descriptor_set(
        &self,
        pipeline: &Arc<ComputePipeline>,
        writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Arc<DescriptorSet> {
        let layout = pipeline.layout().set_layouts()[0].clone();
        DescriptorSet::new(
            self.ccx.descriptor_set_allocator.clone(),
            layout,
            writes,
            [],
        )
        .unwrap()
    }

    /// Submits a one-shot command buffer and blocks until complete.
    fn submit_one_shot(
        &self,
        builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        let cb = builder
            .build()
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cb)
            .map_err(|e| VernierError::Backend(e.to_string()))?
            .then_signal_fence_and_flush()
            .map_err(|e| VernierError::Backend(e.to_string()))?
            .wait(None)
            .map_err(|e| VernierError::Backend(e.to_string()))
    }

    fn new_builder(&self) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| VernierError::Backend(e.to_string()))
    }
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for GpuBackend {
    type Buffer2D = GpuBuffer;
    type Job<'a> = GpuJob<'a>;

    fn begin(&self) -> Result<GpuJob<'_>> {
        let builder = self.new_builder()?;
        Ok(GpuJob { backend: self, builder })
    }

    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<GpuBuffer> {
        if !layout.is_contiguous() {
            return Err(VernierError::NonContiguous {
                stride: layout.row_stride,
                width: layout.width,
            });
        }

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
            data.iter().copied(),
        )
        .map_err(|e| VernierError::Backend(e.to_string()))?;

        let n = layout.width * layout.height;
        let device_buffer = self.alloc_device_buffer((n as u64).next_power_of_two() as usize);

        let mut builder = self.new_builder()?;
        builder
            .copy_buffer(CopyBufferInfo::buffers(host_buffer, device_buffer.clone()))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        self.submit_one_shot(builder)?;

        Ok(GpuBuffer {
            buffer: device_buffer,
            width: layout.width,
            height: layout.height,
        })
    }

    fn download(&self, buffer: &GpuBuffer) -> Result<Vec<Complex32>> {
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
        .map_err(|e| VernierError::Backend(e.to_string()))?;

        let mut builder = self.new_builder()?;
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                buffer.buffer.clone(),
                host_buffer.clone(),
            ))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        self.submit_one_shot(builder)?;

        Ok(host_buffer
            .read()
            .map_err(|e| VernierError::Backend(e.to_string()))?
            .to_vec())
    }

    fn name(&self) -> &str {
        "gpu-vulkan-fft"
    }
}

// ---------------------------------------------------------------------------
// GpuJob — records commands into an AutoCommandBufferBuilder
// ---------------------------------------------------------------------------

pub struct GpuJob<'a> {
    backend: &'a GpuBackend,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

impl GpuJob<'_> {
    fn ds(
        &self,
        pipeline: &Arc<ComputePipeline>,
        writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Arc<DescriptorSet> {
        self.backend.make_descriptor_set(pipeline, writes)
    }

    fn alloc(&self, n: usize) -> Subbuffer<[Complex32]> {
        self.backend.alloc_device_buffer(n)
    }
}

impl ComputeJob for GpuJob<'_> {
    type Buffer2D = GpuBuffer;

    fn copy_buffer(&mut self, src: &GpuBuffer) -> Result<GpuBuffer> {
        let dst = self.alloc(src.size());
        self.builder
            .copy_buffer(CopyBufferInfo::buffers(src.buffer.clone(), dst.clone()))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        Ok(GpuBuffer {
            buffer: dst,
            width: src.width,
            height: src.height,
        })
    }

    fn fft2d(&mut self, buf: &mut GpuBuffer) -> Result<()> {
        let (w, h) = (buf.width, buf.height);
        if w > 2048 || h > 2048 {
            return Err(VernierError::UnsupportedSize(w, h));
        }
        let ds = self.ds(
            &self.backend.ccx.fft_pipeline,
            [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.fft_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.fft_pipeline.layout().clone(),
                0,
                ds,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.fft_pipeline.layout().clone(),
                0,
                fft_shader::PushConstantData { width: w as u32, height: h as u32, pass: 0 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, h as u32, 1]) }.unwrap();
        self.builder
            .push_constants(
                self.backend.ccx.fft_pipeline.layout().clone(),
                0,
                fft_shader::PushConstantData { width: w as u32, height: h as u32, pass: 1 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([w as u32, 1, 1]) }.unwrap();
        Ok(())
    }

    fn ifft2d(&mut self, buf: &mut GpuBuffer) -> Result<()> {
        let (w, h) = (buf.width, buf.height);
        if w > 2048 || h > 2048 {
            return Err(VernierError::UnsupportedSize(w, h));
        }
        let ds = self.ds(
            &self.backend.ccx.ifft_pipeline,
            [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.ifft_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.ifft_pipeline.layout().clone(),
                0,
                ds,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.ifft_pipeline.layout().clone(),
                0,
                ifft_shader::PushConstantData { width: w as u32, height: h as u32, pass: 0 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, h as u32, 1]) }.unwrap();
        self.builder
            .push_constants(
                self.backend.ccx.ifft_pipeline.layout().clone(),
                0,
                ifft_shader::PushConstantData { width: w as u32, height: h as u32, pass: 1 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([w as u32, 1, 1]) }.unwrap();
        Ok(())
    }

    fn extract_phase(&mut self, buf: &GpuBuffer) -> Result<GpuBuffer> {
        let out = self.alloc(buf.size());
        let (w, h) = (buf.width as u32, buf.height as u32);
        let ds = self.ds(
            &self.backend.ccx.extract_phase_pipeline,
            [
                WriteDescriptorSet::buffer(0, buf.buffer.clone()),
                WriteDescriptorSet::buffer(1, out.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.extract_phase_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.extract_phase_pipeline.layout().clone(),
                0,
                ds,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.extract_phase_pipeline.layout().clone(),
                0,
                extract_phase_shader::PushConstantData { width: w, height: h },
            )
            .unwrap();
        unsafe { self.builder.dispatch([(w + 7) / 8, (h + 7) / 8, 1]) }.unwrap();
        Ok(GpuBuffer { buffer: out, width: buf.width, height: buf.height })
    }

    fn filter(
        &mut self,
        buf: &mut GpuBuffer,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()> {
        let (w, h) = (buf.width as u32, buf.height as u32);
        let ds = self.ds(
            &self.backend.ccx.filter_pipeline,
            [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.filter_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.filter_pipeline.layout().clone(),
                0,
                ds,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.filter_pipeline.layout().clone(),
                0,
                filter_shader::PushConstantData {
                    width: w,
                    height: h,
                    min_frequency: min_frequency as u32,
                    max_frequency: max_frequency as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([(w + 7) / 8, (h + 7) / 8, 1]) }.unwrap();
        Ok(())
    }

    fn gaussian_blur_2d(&mut self, buf: &mut GpuBuffer, sigma: Real) -> Result<()> {
        let tmp = self.alloc(buf.size());
        let (w, h) = (buf.width as u32, buf.height as u32);

        let ds_h = self.ds(
            &self.backend.ccx.blur_pipeline,
            [
                WriteDescriptorSet::buffer(0, buf.buffer.clone()),
                WriteDescriptorSet::buffer(1, tmp.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.blur_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.blur_pipeline.layout().clone(),
                0,
                ds_h,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.blur_pipeline.layout().clone(),
                0,
                blur_shader::PushConstantData { width: w, height: h, sigma: sigma as f32, pass: 0 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([(w + 7) / 8, (h + 7) / 8, 1]) }.unwrap();

        let ds_v = self.ds(
            &self.backend.ccx.blur_pipeline,
            [
                WriteDescriptorSet::buffer(0, tmp.clone()),
                WriteDescriptorSet::buffer(1, buf.buffer.clone()),
            ],
        );
        self.builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.blur_pipeline.layout().clone(),
                0,
                ds_v,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.blur_pipeline.layout().clone(),
                0,
                blur_shader::PushConstantData { width: w, height: h, sigma: sigma as f32, pass: 1 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([(w + 7) / 8, (h + 7) / 8, 1]) }.unwrap();
        Ok(())
    }

    fn bandpass_filter(
        &mut self,
        buf: &mut GpuBuffer,
        cx: usize,
        cy: usize,
        sigma: Real,
    ) -> Result<()> {
        let (w, h) = (buf.width as u32, buf.height as u32);
        let ds = self.ds(
            &self.backend.ccx.bandpass_pipeline,
            [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.bandpass_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.bandpass_pipeline.layout().clone(),
                0,
                ds,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.bandpass_pipeline.layout().clone(),
                0,
                bandpass_shader::PushConstantData {
                    width: w,
                    height: h,
                    cx: cx as u32,
                    cy: cy as u32,
                    sigma: sigma as f32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([(w + 7) / 8, (h + 7) / 8, 1]) }.unwrap();
        Ok(())
    }

    fn peak_search(
        &mut self,
        buffer: &mut GpuBuffer,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<GpuBuffer>> {
        let (w, h) = (buffer.width, buffer.height);
        let n = w * h;
        let n_groups = (n + 255) / 256;

        // 1. Deep copy input → working magnitude buffer
        let mag_buf = self.alloc(n);
        self.builder
            .copy_buffer(CopyBufferInfo::buffers(buffer.buffer.clone(), mag_buf.clone()))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        let mut mag = GpuBuffer { buffer: mag_buf, width: w, height: h };

        // 2. magnitude
        let ds_mag = self.ds(
            &self.backend.ccx.magnitude_pipeline,
            [WriteDescriptorSet::buffer(0, mag.buffer.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.magnitude_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.magnitude_pipeline.layout().clone(),
                0,
                ds_mag,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.magnitude_pipeline.layout().clone(),
                0,
                magnitude_shader::PushConstantData { n: n as u32 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([(n as u32 + 255) / 256, 1, 1]) }.unwrap();

        // 3. annulus mask
        self.filter(&mut mag, min_frequency, max_frequency)?;

        // 4. blur
        if smoothing_sigma > 0.0 {
            self.gaussian_blur_2d(&mut mag, smoothing_sigma)?;
        }

        // 5. argmax local → intermediate1 → argmax global → peak1
        let intermediate1 = self.alloc(n_groups);
        let ds_al1 = self.ds(
            &self.backend.ccx.argmax_local_pipeline,
            [
                WriteDescriptorSet::buffer(0, mag.buffer.clone()),
                WriteDescriptorSet::buffer(1, intermediate1.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.argmax_local_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.argmax_local_pipeline.layout().clone(),
                0,
                ds_al1,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.argmax_local_pipeline.layout().clone(),
                0,
                argmax_local_shader::PushConstantData {
                    width: w as u32,
                    height: h as u32,
                    n: n as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([n_groups as u32, 1, 1]) }.unwrap();

        let peak1_buf = self.alloc(1);
        let ds_ag1 = self.ds(
            &self.backend.ccx.argmax_global_pipeline,
            [
                WriteDescriptorSet::buffer(0, intermediate1.clone()),
                WriteDescriptorSet::buffer(1, peak1_buf.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.argmax_global_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.argmax_global_pipeline.layout().clone(),
                0,
                ds_ag1,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.argmax_global_pipeline.layout().clone(),
                0,
                argmax_global_shader::PushConstantData {
                    n_groups: n_groups as u32,
                    width: w as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, 1, 1]) }.unwrap();

        // 6. band + angular filter
        let ds_baf = self.ds(
            &self.backend.ccx.band_angular_filter_pipeline,
            [
                WriteDescriptorSet::buffer(0, mag.buffer.clone()),
                WriteDescriptorSet::buffer(1, peak1_buf.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.band_angular_filter_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.band_angular_filter_pipeline.layout().clone(),
                0,
                ds_baf,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.band_angular_filter_pipeline.layout().clone(),
                0,
                band_angular_filter_shader::PushConstantData {
                    width: w as u32,
                    height: h as u32,
                    sigma: sigma as f32,
                },
            )
            .unwrap();
        unsafe {
            self.builder
                .dispatch([(w as u32 + 7) / 8, (h as u32 + 7) / 8, 1])
        }
        .unwrap();

        // 7. second argmax → peak2
        let intermediate2 = self.alloc(n_groups);
        let ds_al2 = self.ds(
            &self.backend.ccx.argmax_local_pipeline,
            [
                WriteDescriptorSet::buffer(0, mag.buffer.clone()),
                WriteDescriptorSet::buffer(1, intermediate2.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.argmax_local_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.argmax_local_pipeline.layout().clone(),
                0,
                ds_al2,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.argmax_local_pipeline.layout().clone(),
                0,
                argmax_local_shader::PushConstantData {
                    width: w as u32,
                    height: h as u32,
                    n: n as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([n_groups as u32, 1, 1]) }.unwrap();

        let peak2_buf = self.alloc(1);
        let ds_ag2 = self.ds(
            &self.backend.ccx.argmax_global_pipeline,
            [
                WriteDescriptorSet::buffer(0, intermediate2.clone()),
                WriteDescriptorSet::buffer(1, peak2_buf.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.argmax_global_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.argmax_global_pipeline.layout().clone(),
                0,
                ds_ag2,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.argmax_global_pipeline.layout().clone(),
                0,
                argmax_global_shader::PushConstantData {
                    n_groups: n_groups as u32,
                    width: w as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, 1, 1]) }.unwrap();

        // 8. order peaks → 4-element result buffer
        let result_buf = self.alloc(4);
        let ds_ord = self.ds(
            &self.backend.ccx.order_peaks_pipeline,
            [
                WriteDescriptorSet::buffer(0, peak1_buf.clone()),
                WriteDescriptorSet::buffer(1, peak2_buf.clone()),
                WriteDescriptorSet::buffer(2, result_buf.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.order_peaks_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.order_peaks_pipeline.layout().clone(),
                0,
                ds_ord,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.order_peaks_pipeline.layout().clone(),
                0,
                peak_search_shader::PushConstantData { width: w as u32, height: h as u32 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, 1, 1]) }.unwrap();

        Ok(Some(GpuBuffer { buffer: result_buf, width: 2, height: 2 }))
    }

    fn submit(self) -> Result<()> {
        self.backend.submit_one_shot(self.builder)
    }
}

// ---------------------------------------------------------------------------
// Shader modules
// ---------------------------------------------------------------------------

mod fft_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/fft.glsl", include: ["."], spirv_version: "1.3" }
}
mod ifft_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/ifft.glsl", include: ["."], spirv_version: "1.3" }
}
mod blur_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/blur.glsl", include: ["."], spirv_version: "1.3" }
}
mod extract_phase_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/extract_phase.glsl", include: ["."], spirv_version: "1.3" }
}
mod filter_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/filter.glsl", include: ["."], spirv_version: "1.3" }
}
mod peak_search_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/peak_search.glsl", include: ["."], spirv_version: "1.3" }
}
mod bandpass_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/bandpass.glsl", include: ["."], spirv_version: "1.3" }
}
mod magnitude_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/magnitude.glsl", include: ["."], spirv_version: "1.3" }
}
mod argmax_local_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/argmax_local.glsl", include: ["."], spirv_version: "1.3" }
}
mod argmax_global_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/argmax_global.glsl", include: ["."], spirv_version: "1.3" }
}
mod band_angular_filter_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/band_angular_filter.glsl", include: ["."], spirv_version: "1.3" }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-4);
        }
    }

    #[test]
    fn non_square_round_trips() {
        let backend = GpuBackend::new();
        let (data, layout) = checkerboard(16, 4);
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
        }
    }

    #[test]
    fn non_power_of_two_height_round_trips() {
        // width=8 is PoT (uses FFT), height=6 is not (uses direct DFT).
        let backend = GpuBackend::new();
        let (data, layout) = checkerboard(8, 6);
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-3);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-3);
        }
    }
}
