//! Zarr stores (format 2 and 3) and Icechunk repositories, on disk or in object storage.
//!
//! [`ZarrSource`] opens a group (or a single array) as an [`ArraySource`], so Zarr reads like
//! NetCDF: named variables on named dimensions, decoded by [`array::read`](crate::array::read) as
//! xarray decodes them. Reads fetch and decode only the chunks a selection touches, many at once.
//!
//! [`ZarrArray`] is one array read chunk by chunk, for views that schedule their own requests.
//! Its reads are async and run on the caller's runtime, so a view can keep many chunk requests in
//! flight and cancel the ones it no longer needs.

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use zarrs::array::{Array, ArrayMetadata, ArrayMetadataV2, ArraySubset};
use zarrs::metadata::NodeMetadata;
use zarrs::node::{NodePath, meta_key_v2_array, meta_key_v2_attributes, meta_key_v3};
use zarrs::plugin::ExtensionName;
use zarrs::storage::{AsyncReadableListableStorageTraits, AsyncReadableStorageTraits, StoreKey};
use zarrs_object_store::AsyncObjectStore;
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_object_store::object_store::prefix::PrefixStore;
use zarrs_object_store::object_store::{ObjectStore, parse_url_opts};

use crate::array::{ArraySource, AttrValue, Attrs, Variable};
use crate::store::{RetryPolicy, RetryStore, http_store};
use crate::{Error, Result};

type Store = AsyncObjectStore<Box<dyn ObjectStore>>;

/// One array in a Zarr store (format 2 or 3), on disk or reached over HTTPS.
#[derive(Clone)]
pub struct ZarrArray {
    array: Arc<Array<Store>>,
}

impl std::fmt::Debug for ZarrArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrArray")
            .field("shape", &self.shape())
            .field("chunks", &self.chunk_grid())
            .finish()
    }
}

impl ZarrArray {
    /// Opens the array at `path` (such as `/2m_temperature`) in the store at `url`, reading only
    /// its metadata. Reads are retried as [`RetryStore`] describes.
    pub async fn open_http(url: &str, path: &str) -> Result<Self> {
        let http = http_store(url).map_err(zarr_err)?;
        let retry = RetryStore::new(http, RetryPolicy::default());
        Self::open(Box::new(retry), path).await
    }

    /// Opens the array at `path` in any object store, such as a [`RetryStore`] over a store with
    /// faults injected.
    pub async fn open_store(store: Box<dyn ObjectStore>, path: &str) -> Result<Self> {
        Self::open(store, path).await
    }

    /// Opens the array at `path` in a store directory on disk. Read-only: nothing is written.
    pub async fn open_dir(dir: impl AsRef<std::path::Path>, path: &str) -> Result<Self> {
        let fs = LocalFileSystem::new_with_prefix(dir).map_err(zarr_err)?;
        Self::open(Box::new(fs), path).await
    }

    async fn open(store: Box<dyn ObjectStore>, path: &str) -> Result<Self> {
        let store = Arc::new(AsyncObjectStore::new(store));
        // The store's Zarr format isn't known until a metadata key answers, so ask for all three
        // at once: in turn, they cost two more round trips on every open (about 200 ms on a cold
        // connection to Google Cloud Storage).
        let node = NodePath::new(path).map_err(zarr_err)?;
        let keys = [
            meta_key_v3(&node),
            meta_key_v2_array(&node),
            meta_key_v2_attributes(&node),
        ];
        let (v3, v2, attrs) = tokio::join!(
            store.get(&keys[0]),
            store.get(&keys[1]),
            store.get(&keys[2]),
        );
        let metadata = if let Some(json) = v3.map_err(zarr_err)? {
            ArrayMetadata::V3(serde_json::from_slice(&json).map_err(zarr_err)?)
        } else if let Some(json) = v2.map_err(zarr_err)? {
            let mut m: ArrayMetadataV2 = serde_json::from_slice(&json).map_err(zarr_err)?;
            if let Some(json) = attrs.map_err(zarr_err)? {
                m.attributes = serde_json::from_slice(&json).map_err(zarr_err)?;
            }
            ArrayMetadata::V2(m)
        } else {
            return Err(Error::Zarr(format!("no Zarr array at {path}")));
        };
        let array = Array::new_with_metadata(store, path, metadata).map_err(zarr_err)?;
        Ok(Self {
            array: Arc::new(array),
        })
    }

    pub fn shape(&self) -> &[u64] {
        self.array.shape()
    }

    /// Chunks along each dimension.
    pub fn chunk_grid(&self) -> &[u64] {
        self.array.chunk_grid_shape()
    }

    /// Reads and decodes one chunk as `f32`. Chunks missing from the store come back as the fill
    /// value.
    pub async fn chunk_f32(&self, indices: &[u64]) -> Result<Vec<f32>> {
        self.array
            .async_retrieve_chunk::<Vec<f32>>(indices)
            .await
            .map_err(zarr_err)
    }

    /// Reads one chunk and returns only `ranges` of it (relative to the chunk), as `f32` in C
    /// order. The whole chunk is still fetched and, for most codecs, decoded; only the subset is
    /// kept.
    pub async fn chunk_subset_f32(
        &self,
        indices: &[u64],
        ranges: &[Range<u64>],
    ) -> Result<Vec<f32>> {
        self.array
            .async_retrieve_chunk_subset::<Vec<f32>>(indices, &ArraySubset::new_with_ranges(ranges))
            .await
            .map_err(zarr_err)
    }
}

fn zarr_err(e: impl std::fmt::Display) -> Error {
    Error::Zarr(e.to_string())
}

type DynStore = dyn AsyncReadableListableStorageTraits;

/// A Zarr group, or one array, opened as an [`ArraySource`].
///
/// Dimension names come from `dimension_names` (format 3) or xarray's `_ARRAY_DIMENSIONS`
/// attribute (format 2); arrays with neither get `<name>_dim_N`. CF decoding follows xarray:
/// in format 2 the array's `fill_value` is the `_FillValue`, while in format 3 only a
/// `_FillValue` attribute marks missing values (the fill value is what unwritten chunks hold).
///
/// Opening reads only metadata: consolidated metadata when the store has it (`.zmetadata` in
/// format 2, `consolidated_metadata` in format 3), otherwise one key per array found by listing
/// the group. Reads run on a runtime of their own, so [`read_raw`](ArraySource::read_raw) may be
/// called from any thread, including one inside another runtime, and connections a source opens
/// are always driven by the runtime that opened them.
pub struct ZarrSource {
    arrays: Vec<Arc<Array<DynStore>>>,
    vars: Vec<Variable>,
    attrs: Attrs,
}

impl std::fmt::Debug for ZarrSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrSource")
            .field("variables", &self.vars)
            .finish_non_exhaustive()
    }
}

/// Which version of an Icechunk repository to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IcechunkVersion {
    /// The tip of a branch, such as `main`.
    Branch(String),
    Tag(String),
    /// A snapshot ID, as Icechunk prints it (20 base-32 characters).
    Snapshot(String),
}

impl ZarrSource {
    /// Opens group `group` (`/` for the root) of the Zarr store at `location`: a directory, or
    /// an `https://`, `s3://`, `gs://` or `az://` URL. Cloud credentials and regions come from the
    /// environment (`AWS_*`, `GOOGLE_*`, `AZURE_*`) as each provider's own tools read them. Reads
    /// over the network are retried as [`RetryStore`] describes.
    pub fn open(location: &str, group: &str) -> Result<Self> {
        Self::open_store(object_store_at(location)?, group)
    }

    /// Opens group `group` of a Zarr store in a directory. Read-only: nothing is written.
    pub fn open_dir(dir: impl AsRef<std::path::Path>, group: &str) -> Result<Self> {
        let fs = LocalFileSystem::new_with_prefix(dir).map_err(zarr_err)?;
        Self::open_store(Box::new(fs), group)
    }

    /// Opens group `group` of a Zarr store in any object store, such as one built with the
    /// caller's own credentials.
    pub fn open_store(store: Box<dyn ObjectStore>, group: &str) -> Result<Self> {
        let store: Arc<DynStore> = Arc::new(AsyncObjectStore::new(store));
        let group = group.to_string();
        block_on(async move { Self::from_storage(store, &group).await })
    }

    /// Opens group `group` of the Icechunk repository at `location` (a directory, or an `s3://`,
    /// `gs://` or `https://` URL) as it was at `version`. The session is read-only: nothing is
    /// ever committed. Needs the `icechunk` feature.
    pub fn open_icechunk(location: &str, version: &IcechunkVersion, group: &str) -> Result<Self> {
        let (location, version, group) = (location.to_string(), version.clone(), group.to_string());
        block_on(async move {
            let store = icechunk_repo::session_store(&location, &version).await?;
            Self::from_storage(store, &group).await
        })
    }

    /// The branches and tags of the Icechunk repository at `location`, sorted. Needs the
    /// `icechunk` feature.
    pub fn icechunk_refs(location: &str) -> Result<(Vec<String>, Vec<String>)> {
        let location = location.to_string();
        block_on(async move { icechunk_repo::refs(&location).await })
    }

    /// Like [`read_raw`](ArraySource::read_raw), for async callers: the read still runs on the
    /// Zarr runtime, and this waits for it without blocking a thread.
    pub async fn read_raw_async(
        &self,
        var: &str,
        start: &[u64],
        count: &[u64],
    ) -> Result<Vec<f64>> {
        let (array, subset) = self.selection(var, start, count)?;
        runtime()
            .spawn(read_f64(array, subset))
            .await
            .map_err(zarr_err)?
    }

    /// The chunk shape of `var`, for callers that align their reads to chunks.
    pub fn chunk_shape(&self, var: &str) -> Option<Vec<u64>> {
        let i = self.vars.iter().position(|v| v.name == var)?;
        let origin = vec![0; self.vars[i].shape.len()];
        let shape = self.arrays[i].chunk_shape(&origin).ok()?;
        Some(shape.iter().map(|n| n.get()).collect())
    }

    async fn from_storage(store: Arc<DynStore>, group: &str) -> Result<Self> {
        let node = NodePath::new(group).map_err(zarr_err)?;
        // As in ZarrArray::open, ask for every metadata key at once rather than one at a time.
        let (v3, consolidated, zgroup, zarray, zattrs) = tokio::join!(
            get_json(&store, &node, "zarr.json"),
            get_json(&store, &node, ".zmetadata"),
            get_json(&store, &node, ".zgroup"),
            get_json(&store, &node, ".zarray"),
            get_json(&store, &node, ".zattrs"),
        );
        let (v3, consolidated, zgroup, zarray) = (v3?, consolidated?, zgroup?, zarray?);
        let zattrs = as_map(zattrs?.as_ref());
        // (name, path, metadata) of each array.
        let mut found: Vec<(String, NodePath, ArrayMetadata)> = Vec::new();
        let leaf = || {
            node.as_str()
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_string()
        };
        let child = |name: &str| NodePath::new(&join(&node, name)).map_err(zarr_err);
        let attrs;
        if let Some(meta) = v3 {
            if meta["node_type"] == "array" {
                let m = serde_json::from_value(meta).map_err(zarr_err)?;
                found.push((leaf(), node.clone(), ArrayMetadata::V3(m)));
                attrs = serde_json::Map::new();
            } else {
                attrs = as_map(meta.get("attributes"));
                match meta
                    .pointer("/consolidated_metadata/metadata")
                    .and_then(|m| m.as_object())
                {
                    Some(nodes) => {
                        for (name, m) in nodes {
                            if !name.contains('/') && m["node_type"] == "array" {
                                let m = serde_json::from_value(m.clone()).map_err(zarr_err)?;
                                found.push((name.clone(), child(name)?, ArrayMetadata::V3(m)));
                            }
                        }
                    }
                    None => found = list_arrays(&store, &node).await?,
                }
            }
        } else if let Some(meta) = zarray {
            let mut m: ArrayMetadataV2 = serde_json::from_value(meta).map_err(zarr_err)?;
            m.attributes = zattrs;
            found.push((leaf(), node.clone(), ArrayMetadata::V2(m)));
            attrs = serde_json::Map::new();
        } else if let Some(meta) = consolidated {
            let nodes = as_map(meta.get("metadata"));
            attrs = as_map(nodes.get(".zattrs"));
            for (key, m) in &nodes {
                let Some(name) = key.strip_suffix("/.zarray") else {
                    continue;
                };
                if name.contains('/') {
                    continue;
                }
                let mut m: ArrayMetadataV2 = serde_json::from_value(m.clone()).map_err(zarr_err)?;
                m.attributes = as_map(nodes.get(&format!("{name}/.zattrs")));
                found.push((name.to_string(), child(name)?, ArrayMetadata::V2(m)));
            }
        } else if zgroup.is_some() {
            attrs = zattrs;
            found = list_arrays(&store, &node).await?;
        } else {
            return Err(Error::Zarr(format!("no Zarr group or array at {group}")));
        }
        found.sort_by(|a, b| a.0.cmp(&b.0));

        let mut arrays = Vec::with_capacity(found.len());
        let mut vars = Vec::with_capacity(found.len());
        for (name, path, metadata) in found {
            let var = variable(&name, &metadata);
            let array = Array::new_with_metadata(store.clone(), path.as_str(), metadata)
                .map_err(zarr_err)?;
            vars.push(Variable {
                dtype: numpy_dtype(&array),
                ..var
            });
            arrays.push(Arc::new(array));
        }
        Ok(Self {
            arrays,
            vars,
            attrs: json_attrs(&attrs),
        })
    }

    fn selection(
        &self,
        var: &str,
        start: &[u64],
        count: &[u64],
    ) -> Result<(Arc<Array<DynStore>>, ArraySubset)> {
        let i = self
            .vars
            .iter()
            .position(|v| v.name == var)
            .ok_or_else(|| Error::Unsupported(format!("no variable {var:?}")))?;
        crate::array::check_slab(&self.vars[i].shape, start, count)?;
        let subset =
            ArraySubset::new_with_start_shape(start.to_vec(), count.to_vec()).map_err(zarr_err)?;
        Ok((self.arrays[i].clone(), subset))
    }
}

impl ArraySource for ZarrSource {
    fn variables(&self) -> &[Variable] {
        &self.vars
    }

    fn attrs(&self) -> &Attrs {
        &self.attrs
    }

    fn read_raw(&self, var: &str, start: &[u64], count: &[u64]) -> Result<Vec<f64>> {
        let (array, subset) = self.selection(var, start, count)?;
        block_on(read_f64(array, subset))
    }
}

/// The runtime every [`ZarrSource`] opens and reads on.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .thread_name("dv-zarr")
            .enable_all()
            .build()
            .expect("the Zarr runtime starts")
    })
}

/// Runs `fut` on the Zarr runtime and blocks this thread until it finishes. Unlike
/// `Runtime::block_on`, this works inside another runtime too.
fn block_on<T: Send + 'static>(fut: impl Future<Output = Result<T>> + Send + 'static) -> Result<T> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    runtime().spawn(async move {
        let _ = tx.send(fut.await);
    });
    rx.recv()
        .map_err(|_| Error::Zarr("a Zarr task panicked".into()))?
}

async fn read_f64(array: Arc<Array<DynStore>>, subset: ArraySubset) -> Result<Vec<f64>> {
    macro_rules! read {
        ($t:ty) => {
            array
                .async_retrieve_array_subset::<Vec<$t>>(&subset)
                .await
                .map_err(zarr_err)?
                .into_iter()
                .map(|x| x as f64)
                .collect()
        };
    }
    let name = array.data_type().name_v3().unwrap_or_default();
    Ok(match name.as_ref() {
        "int8" => read!(i8),
        "int16" => read!(i16),
        "int32" => read!(i32),
        "int64" => read!(i64),
        "uint8" => read!(u8),
        "uint16" => read!(u16),
        "uint32" => read!(u32),
        "uint64" => read!(u64),
        "float32" => read!(f32),
        "float64" => read!(f64),
        "bool" => array
            .async_retrieve_array_subset::<Vec<bool>>(&subset)
            .await
            .map_err(zarr_err)?
            .into_iter()
            .map(|b| f64::from(u8::from(b)))
            .collect(),
        _ => {
            return Err(Error::Unsupported(format!(
                "Zarr arrays of {} can't be read as numbers",
                array.data_type()
            )));
        }
    })
}

/// A variable's name, dimensions and attributes from its metadata. Its dtype is filled in from
/// the opened array.
fn variable(name: &str, metadata: &ArrayMetadata) -> Variable {
    let (shape, mut attrs, dims) = match metadata {
        ArrayMetadata::V3(m) => {
            let dims = serde_json::to_value(&m.dimension_names).unwrap_or_default();
            (m.shape.clone(), m.attributes.clone(), dims)
        }
        ArrayMetadata::V2(m) => {
            let mut attrs = m.attributes.clone();
            let dims = attrs.remove("_ARRAY_DIMENSIONS").unwrap_or_default();
            (m.shape.clone(), attrs, dims)
        }
    };
    let dims = (0..shape.len())
        .map(|i| match dims.get(i).and_then(|d| d.as_str()) {
            Some(d) => d.to_string(),
            None => format!("{name}_dim_{i}"),
        })
        .collect();
    if let ArrayMetadata::V2(m) = metadata {
        let fill = serde_json::to_value(&m.fill_value).unwrap_or_default();
        if let Some(fill) = json_number(&fill)
            && !attrs.contains_key("_FillValue")
        {
            attrs.insert("_FillValue".into(), fill.into());
        }
    }
    Variable {
        name: name.to_string(),
        dims,
        shape,
        dtype: String::new(),
        attrs: json_attrs(&attrs),
    }
}

/// The array's element type in NumPy's notation (`i2`, `f4`…), or its Zarr name for types NumPy
/// spells differently.
fn numpy_dtype(array: &Array<DynStore>) -> String {
    let name = array.data_type().name_v3().unwrap_or_default();
    match name.as_ref() {
        "bool" => "b1".into(),
        "int8" => "i1".into(),
        "int16" => "i2".into(),
        "int32" => "i4".into(),
        "int64" => "i8".into(),
        "uint8" => "u1".into(),
        "uint16" => "u2".into(),
        "uint32" => "u4".into(),
        "uint64" => "u8".into(),
        "float16" => "f2".into(),
        "float32" => "f4".into(),
        "float64" => "f8".into(),
        _ => array.data_type().to_string(),
    }
}

/// A number, or one of the strings Zarr and xarray write for non-finite floats.
fn json_number(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::Bool(b) => Some(f64::from(u8::from(*b))),
        serde_json::Value::String(s) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

fn json_attrs(map: &serde_json::Map<String, serde_json::Value>) -> Attrs {
    use serde_json::Value;
    let numeric_keys = ["_FillValue", "missing_value", "valid_min", "valid_max"];
    map.iter()
        .filter_map(|(k, v)| {
            let value = match v {
                Value::String(s) if numeric_keys.contains(&k.as_str()) => match json_number(v) {
                    Some(x) => AttrValue::Numbers(vec![x]),
                    None => AttrValue::Text(s.clone()),
                },
                Value::String(s) => AttrValue::Text(s.clone()),
                Value::Number(_) | Value::Bool(_) => AttrValue::Numbers(vec![json_number(v)?]),
                Value::Array(items) if items.iter().all(|x| x.is_string()) => AttrValue::Text(
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join("\n"),
                ),
                Value::Array(items) => {
                    AttrValue::Numbers(items.iter().map(json_number).collect::<Option<_>>()?)
                }
                Value::Null | Value::Object(_) => return None,
            };
            Some((k.clone(), value))
        })
        .collect()
}

fn as_map(v: Option<&serde_json::Value>) -> serde_json::Map<String, serde_json::Value> {
    v.and_then(|v| v.as_object()).cloned().unwrap_or_default()
}

/// The path of `name` inside the group at `node`, without the leading `/` store keys don't have.
fn join(node: &NodePath, name: &str) -> String {
    let base = node.as_str().trim_end_matches('/');
    format!("{base}/{name}")
}

async fn get_json(
    store: &Arc<DynStore>,
    node: &NodePath,
    name: &str,
) -> Result<Option<serde_json::Value>> {
    let key = StoreKey::new(join(node, name).trim_start_matches('/')).map_err(zarr_err)?;
    match store.get(&key).await.map_err(zarr_err)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes).map_err(zarr_err)?)),
        None => Ok(None),
    }
}

/// The arrays directly in a group without consolidated metadata, found by listing the store.
async fn list_arrays(
    store: &Arc<DynStore>,
    node: &NodePath,
) -> Result<Vec<(String, NodePath, ArrayMetadata)>> {
    let children = zarrs::node::async_get_child_nodes(store, node, false)
        .await
        .map_err(|e| {
            Error::Zarr(format!(
                "listing the group {node} failed, and it has no consolidated metadata: {e}"
            ))
        })?;
    Ok(children
        .into_iter()
        .filter_map(|c| match c.metadata() {
            NodeMetadata::Array(m) => {
                Some((c.name().as_str().to_string(), c.path().clone(), m.clone()))
            }
            NodeMetadata::Group(_) => None,
        })
        .collect())
}

/// The object store at `location`: a directory, or an `http(s)://`, `file://` or cloud URL.
fn object_store_at(location: &str) -> Result<Box<dyn ObjectStore>> {
    let Some((scheme, _)) = location.split_once("://") else {
        return Ok(Box::new(
            LocalFileSystem::new_with_prefix(location).map_err(zarr_err)?,
        ));
    };
    let url = url::Url::parse(location).map_err(zarr_err)?;
    let store: Box<dyn ObjectStore> = match scheme {
        "file" => {
            let dir = url
                .to_file_path()
                .map_err(|()| Error::Zarr(format!("{location} isn't a local path")))?;
            return Ok(Box::new(
                LocalFileSystem::new_with_prefix(dir).map_err(zarr_err)?,
            ));
        }
        "http" | "https" => http_store(location).map_err(zarr_err)?,
        _ => {
            let (store, prefix) = parse_url_opts(&url, cloud_env()).map_err(zarr_err)?;
            if prefix.as_ref().is_empty() {
                store
            } else {
                Box::new(PrefixStore::new(store, prefix))
            }
        }
    };
    Ok(Box::new(RetryStore::new(store, RetryPolicy::default())))
}

/// Cloud settings from the environment, keyed as object_store's builders read them.
fn cloud_env() -> Vec<(String, String)> {
    std::env::vars()
        .filter(|(k, _)| {
            ["AWS_", "GOOGLE_", "AZURE_"]
                .iter()
                .any(|p| k.starts_with(p))
        })
        .map(|(k, v)| (k.to_ascii_lowercase(), v))
        .collect()
}

#[cfg(feature = "icechunk")]
mod icechunk_repo {
    //! Icechunk repositories through the icechunk crate, read through a read-only session.

    use std::sync::Arc;

    use icechunk::Repository;
    use icechunk::format::SnapshotId;
    use icechunk::repository::VersionInfo;
    use icechunk::storage::{self, S3Credentials, S3Options, Storage};
    use zarrs_icechunk::AsyncIcechunkStore;

    use super::{DynStore, IcechunkVersion, zarr_err};
    use crate::{Error, Result};

    async fn storage_at(location: &str) -> Result<Arc<dyn Storage + Send + Sync>> {
        let Some((scheme, rest)) = location.split_once("://") else {
            return storage::new_local_filesystem_storage(std::path::Path::new(location))
                .await
                .map_err(zarr_err);
        };
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) if !p.trim_matches('/').is_empty() => {
                (b.to_string(), Some(p.trim_matches('/').to_string()))
            }
            Some((b, _)) => (b.to_string(), None),
            None => (rest.to_string(), None),
        };
        let env = |k: &str| std::env::var(k).ok();
        match scheme {
            "http" | "https" => storage::new_http_storage(location, None, None).map_err(zarr_err),
            "s3" => {
                let mut options = S3Options::default();
                options.region = env("AWS_REGION").or_else(|| env("AWS_DEFAULT_REGION"));
                options.endpoint_url = env("AWS_ENDPOINT_URL");
                storage::new_s3_object_store_storage(
                    options,
                    bucket,
                    prefix,
                    Some(S3Credentials::FromEnv),
                    Vec::new(),
                    Vec::new(),
                )
                .await
                .map_err(zarr_err)
            }
            "gs" | "gcs" => {
                storage::new_gcs_storage(bucket, prefix, None, None, Vec::new(), Vec::new())
                    .map_err(zarr_err)
            }
            _ => Err(Error::Unsupported(format!(
                "Icechunk repositories at {scheme}:// URLs aren't supported"
            ))),
        }
    }

    async fn open(location: &str) -> Result<Repository> {
        let storage = storage_at(location).await?;
        Repository::open(storage).execute().await.map_err(zarr_err)
    }

    pub(super) async fn session_store(
        location: &str,
        version: &IcechunkVersion,
    ) -> Result<Arc<DynStore>> {
        let repo = open(location).await?;
        let version = match version {
            IcechunkVersion::Branch(b) => VersionInfo::BranchTipRef(b.clone()),
            IcechunkVersion::Tag(t) => VersionInfo::TagRef(t.clone()),
            IcechunkVersion::Snapshot(id) => VersionInfo::SnapshotId(
                SnapshotId::try_from(id.as_str())
                    .map_err(|e| Error::Zarr(format!("snapshot {id}: {e}")))?,
            ),
        };
        let session = repo.readonly_session(&version).await.map_err(zarr_err)?;
        Ok(Arc::new(AsyncIcechunkStore::new(session)))
    }

    pub(super) async fn refs(location: &str) -> Result<(Vec<String>, Vec<String>)> {
        let repo = open(location).await?;
        let branches = repo.list_branches().await.map_err(zarr_err)?;
        let tags = repo.list_tags().await.map_err(zarr_err)?;
        Ok((branches.into_iter().collect(), tags.into_iter().collect()))
    }
}

#[cfg(not(feature = "icechunk"))]
mod icechunk_repo {
    use std::sync::Arc;

    use super::{DynStore, IcechunkVersion};
    use crate::{Error, Result};

    fn unsupported<T>() -> Result<T> {
        Err(Error::Unsupported(
            "Icechunk repositories need the `icechunk` feature, which this build doesn't include"
                .into(),
        ))
    }

    pub(super) async fn session_store(_: &str, _: &IcechunkVersion) -> Result<Arc<DynStore>> {
        unsupported()
    }

    pub(super) async fn refs(_: &str) -> Result<(Vec<String>, Vec<String>)> {
        unsupported()
    }
}
