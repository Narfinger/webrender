use api::{
    units::{DeviceIntSize, TexelRect},
    ImageBufferKind, ImageFormat,
};

use crate::{
    device::{ExternalTexture, Texture, TextureFilter},
    internal_types::{
        CacheTextureId, DeferredResolveIndex, Swizzle, TextureCacheCategory, TextureSource,
        TextureSourceExternal,
    },
    profiler::{self, TransactionProfile},
    Device, FastHashMap, MemoryReport,
};

use super::TextureSampler;

pub(crate) struct CacheTexture {
    pub(crate) texture: Texture,
    pub(crate) category: TextureCacheCategory,
}

/// Helper struct for resolving device Textures for use during rendering passes.
///
/// Manages the mapping between the at-a-distance texture handles used by the
/// `RenderBackend` (which does not directly interface with the GPU) and actual
/// device texture handles.
pub(crate) struct TextureResolver {
    /// A map to resolve texture cache IDs to native textures.
    texture_cache_map: FastHashMap<CacheTextureId, CacheTexture>,

    /// Map of external image IDs to native textures.
    external_images: FastHashMap<DeferredResolveIndex, ExternalTexture>,

    /// A special 1x1 dummy texture used for shaders that expect to work with
    /// the output of the previous pass but are actually running in the first
    /// pass.
    dummy_cache_texture: Texture,
}

impl TextureResolver {
    pub(crate) fn new(device: &mut Device) -> TextureResolver {
        let dummy_cache_texture = device.create_texture(
            ImageBufferKind::Texture2D,
            ImageFormat::RGBA8,
            1,
            1,
            TextureFilter::Linear,
            None,
        );
        device.upload_texture_immediate(&dummy_cache_texture, &[0xff, 0xff, 0xff, 0xff]);

        TextureResolver {
            texture_cache_map: FastHashMap::default(),
            external_images: FastHashMap::default(),
            dummy_cache_texture,
        }
    }

    pub(crate) fn deinit(self, device: &mut Device) {
        device.delete_texture(self.dummy_cache_texture);

        for (_id, item) in self.texture_cache_map {
            device.delete_texture(item.texture);
        }
    }

    pub(crate) fn begin_frame(&mut self) {}

    pub(crate) fn end_pass(
        &mut self,
        device: &mut Device,
        textures_to_invalidate: &[CacheTextureId],
    ) {
        // For any texture that is no longer needed, immediately
        // invalidate it so that tiled GPUs don't need to resolve it
        // back to memory.
        for texture_id in textures_to_invalidate {
            if let Some(render_target) =
                &self.texture_cache_map.get(texture_id).map(|cm| &cm.texture)
            {
                device.invalidate_render_target(render_target);
            }
        }
    }

    // Bind a source texture to the device.
    pub(crate) fn bind(
        &self,
        texture_id: &TextureSource,
        sampler: TextureSampler,
        device: &mut Device,
    ) -> Swizzle {
        match *texture_id {
            TextureSource::Invalid => Swizzle::default(),
            TextureSource::Dummy => {
                let swizzle = Swizzle::default();
                device.bind_texture(sampler, &self.dummy_cache_texture, swizzle);
                swizzle
            }
            TextureSource::External(TextureSourceExternal { ref index, .. }) => {
                let texture = self
                    .external_images
                    .get(index)
                    .expect("BUG: External image should be resolved by now");
                device.bind_external_texture(sampler, texture);
                Swizzle::default()
            }
            TextureSource::TextureCache(index, swizzle) => {
                if let Some(texture) = self.texture_cache_map.get(&index).map(|cm| &cm.texture) {
                    device.bind_texture(sampler, texture, swizzle);
                }
                swizzle
            }
        }
    }

    // Get the real (OpenGL) texture ID for a given source texture.
    // For a texture cache texture, the IDs are stored in a vector
    // map for fast access.
    pub(crate) fn resolve(&self, texture_id: &TextureSource) -> Option<(&Texture, Swizzle)> {
        match *texture_id {
            TextureSource::Invalid => None,
            TextureSource::Dummy => Some((&self.dummy_cache_texture, Swizzle::default())),
            TextureSource::External(..) => {
                panic!("BUG: External textures cannot be resolved, they can only be bound.");
            }
            TextureSource::TextureCache(index, swizzle) => {
                if let Some(texture) = self.texture_cache_map.get(&index).map(|cm| &cm.texture) {
                    Some((&texture, swizzle))
                } else {
                    None
                }
            }
        }
    }

    // Retrieve the deferred / resolved UV rect if an external texture, otherwise
    // return the default supplied UV rect.
    pub(crate) fn get_uv_rect(
        &self,
        source: &TextureSource,
        default_value: TexelRect,
    ) -> TexelRect {
        match source {
            TextureSource::External(TextureSourceExternal { ref index, .. }) => {
                let texture = self
                    .external_images
                    .get(index)
                    .expect("BUG: External image should be resolved by now");
                texture.get_uv_rect()
            }
            _ => default_value,
        }
    }

    /// Returns the size of the texture in pixels
    pub(crate) fn get_texture_size(&self, texture: &TextureSource) -> DeviceIntSize {
        match *texture {
            TextureSource::Invalid => DeviceIntSize::zero(),
            TextureSource::TextureCache(id, _) => self
                .texture_cache_map
                .get(&id)
                .map(|cm| cm.texture.get_dimensions())
                .unwrap_or(DeviceIntSize::zero()),
            TextureSource::External(TextureSourceExternal { index, .. }) => {
                // If UV coords are normalized then this value will be incorrect. However, the
                // texture size is currently only used to set the uTextureSize uniform, so that
                // shaders without access to textureSize() can normalize unnormalized UVs. Which
                // means this is not a problem.
                let uv_rect = self.external_images[&index].get_uv_rect();
                (uv_rect.uv1 - uv_rect.uv0).abs().to_size().to_i32()
            }
            TextureSource::Dummy => DeviceIntSize::new(1, 1),
        }
    }

    pub(crate) fn report_memory(&self) -> MemoryReport {
        let mut report = MemoryReport::default();

        // We're reporting GPU memory rather than heap-allocations, so we don't
        // use size_of_op.
        for item in self.texture_cache_map.values() {
            let counter = match item.category {
                TextureCacheCategory::Atlas => &mut report.atlas_textures,
                TextureCacheCategory::Standalone => &mut report.standalone_textures,
                TextureCacheCategory::PictureTile => &mut report.picture_tile_textures,
                TextureCacheCategory::RenderTarget => &mut report.render_target_textures,
            };
            *counter += item.texture.size_in_bytes();
        }

        report
    }

    pub(crate) fn update_profile(&self, profile: &mut TransactionProfile) {
        let mut external_image_bytes = 0;
        for img in self.external_images.values() {
            let uv_rect = img.get_uv_rect();
            // If UV coords are normalized then this value will be incorrect. This is unfortunate
            // but doesn't impact end users at all.
            let size = (uv_rect.uv1 - uv_rect.uv0).abs().to_size().to_i32();

            // Assume 4 bytes per pixels which is true most of the time but
            // not always.
            let bpp = 4;
            external_image_bytes += size.area() as usize * bpp;
        }

        profile.set(
            profiler::EXTERNAL_IMAGE_BYTES,
            profiler::bytes_to_mb(external_image_bytes),
        );
    }

    pub(crate) fn get_cache_texture_mut(&mut self, id: &CacheTextureId) -> &mut Texture {
        &mut self
            .texture_cache_map
            .get_mut(id)
            .expect("bug: texture not allocated")
            .texture
    }

    pub(crate) fn get_texture_from_cache(&self, id: &CacheTextureId) -> Option<&Texture> {
        self.texture_cache_map.get(id).map(|ct| &ct.texture)
    }

    pub(crate) fn remove_texture_from_cache(
        &mut self,
        id: &CacheTextureId,
    ) -> Option<CacheTexture> {
        self.texture_cache_map.remove(&id)
    }

    pub(crate) fn insert_texture_into_cache(&mut self, id: CacheTextureId, texture: CacheTexture) {
        self.texture_cache_map.insert(id, texture);
    }

    pub(crate) fn texture_cache(&self) -> &FastHashMap<CacheTextureId, CacheTexture> {
        &self.texture_cache_map
    }

    pub(crate) fn texture_cache_drain(
        &mut self,
    ) -> impl Iterator<Item = (CacheTextureId, CacheTexture)> + use<'_> {
        self.texture_cache_map.drain()
    }

    pub(crate) fn insert_external_image(
        &mut self,
        id: DeferredResolveIndex,
        texture: ExternalTexture,
    ) {
        self.external_images.insert(id, texture);
    }

    pub(crate) fn external_image_drain(
        &mut self,
    ) -> impl Iterator<Item = (DeferredResolveIndex, ExternalTexture)> + use<'_> {
        self.external_images.drain()
    }

    pub(crate) fn external_images(&self) -> &FastHashMap<DeferredResolveIndex, ExternalTexture> {
        &self.external_images
    }
}
