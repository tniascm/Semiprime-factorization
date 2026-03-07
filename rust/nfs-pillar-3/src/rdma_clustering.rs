use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Simulates a registered RDMA memory region (Memory Region / MR)
/// In a real system, this would be a memory buffer registered with the NIC via ibverbs.
pub struct MemoryRegion {
    pub buffer: Vec<u8>,
    pub rkey: u32,
    pub addr: u64,
}

impl MemoryRegion {
    pub fn new(size: usize, rkey: u32, addr: u64) -> Self {
        MemoryRegion {
            buffer: vec![0; size],
            rkey,
            addr,
        }
    }
}

/// Represents a remote node's memory region metadata that allows RDMA ops.
#[derive(Clone, Copy)]
pub struct RemoteRegion {
    pub rkey: u32,
    pub addr: u64,
}

/// Simulates a minimal RDMA network interface card (NIC)
/// facilitating direct memory access without remote CPU involvement.
pub struct SimulatedNic {
    memory_regions: Arc<Mutex<HashMap<u64, Arc<Mutex<MemoryRegion>>>>>,
}

impl Default for SimulatedNic {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatedNic {
    pub fn new() -> Self {
        SimulatedNic {
            memory_regions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn register_mr(&self, mr: Arc<Mutex<MemoryRegion>>) {
        let mut regions = self.memory_regions.lock().unwrap();
        let addr = {
            let mr_lock = mr.lock().unwrap();
            mr_lock.addr
        };
        regions.insert(addr, mr);
    }

    /// Simulates RDMA Write: writes `data` directly to the `remote_addr`.
    /// Does not notify the remote CPU (zero-copy networking).
    pub fn rdma_write(&self, remote_mr: RemoteRegion, offset: usize, data: &[u8]) -> Result<(), String> {
        let regions = self.memory_regions.lock().unwrap();
        if let Some(mr_arc) = regions.get(&remote_mr.addr) {
            let mut mr = mr_arc.lock().unwrap();
            if mr.rkey != remote_mr.rkey {
                return Err("Invalid rkey".to_string());
            }
            if offset + data.len() > mr.buffer.len() {
                return Err("Out of bounds".to_string());
            }
            // Zero-copy network transfer simulated by a local memory copy
            mr.buffer[offset..offset + data.len()].copy_from_slice(data);
            Ok(())
        } else {
            Err("Remote MR not found".to_string())
        }
    }
}

/// A node in the clustering system (could be Master or Worker).
pub struct ClusterNode {
    pub local_mr: Arc<Mutex<MemoryRegion>>,
    pub nic: Arc<SimulatedNic>,
}

impl ClusterNode {
    pub fn new(nic: Arc<SimulatedNic>, mr_size: usize, addr: u64, rkey: u32) -> Self {
        let mr = Arc::new(Mutex::new(MemoryRegion::new(mr_size, rkey, addr)));
        nic.register_mr(Arc::clone(&mr));
        ClusterNode { local_mr: mr, nic }
    }

    pub fn get_remote_region_info(&self) -> RemoteRegion {
        let mr = self.local_mr.lock().unwrap();
        RemoteRegion { rkey: mr.rkey, addr: mr.addr }
    }

    /// Stream an array of 64-bit relations directly into the master's memory.
    /// In the NFS pipeline, this happens synchronously and lock-free across the PCIe bus.
    pub fn stream_relations_to_master(&self, master_region: RemoteRegion, offset_bytes: usize, relations: &[u64]) -> Result<(), String> {
        // Serialize to bytes
        let mut data = Vec::with_capacity(relations.len() * 8);
        for rel in relations {
            data.extend_from_slice(&rel.to_ne_bytes());
        }
        self.nic.rdma_write(master_region, offset_bytes, &data)
    }

    /// Read raw relation bytes from the local RDMA registered region.
    pub fn read_local_relations(&self, count: usize) -> Vec<u64> {
        let mr = self.local_mr.lock().unwrap();
        let mut rels = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * 8;
            let end = start + 8;
            if end > mr.buffer.len() {
                break;
            }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&mr.buffer[start..end]);
            rels.push(u64::from_ne_bytes(buf));
        }
        rels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_rdma_pipeline() {
        let nic = Arc::new(SimulatedNic::new());

        // Master node allocates 1024 bytes (128 relations)
        let master = ClusterNode::new(Arc::clone(&nic), 1024, 0x1000, 42);
        let master_region = master.get_remote_region_info();

        // Worker node does not need a large MR for receiving
        let worker1 = ClusterNode::new(Arc::clone(&nic), 8, 0x2000, 43);
        let worker2 = ClusterNode::new(Arc::clone(&nic), 8, 0x3000, 44);

        let w1_rels: Vec<u64> = vec![1, 2, 3];
        let w2_rels: Vec<u64> = vec![99, 100];

        // Workers perform RDMA write directly into Master's memory at specific offsets
        worker1.stream_relations_to_master(master_region, 0, &w1_rels).unwrap();

        // Worker 2 writes after Worker 1's data (offset 24 bytes)
        worker2.stream_relations_to_master(master_region, 24, &w2_rels).unwrap();

        // Master reads from its local memory without parsing files or blocking on channels
        let read_rels = master.read_local_relations(5);
        assert_eq!(read_rels, vec![1, 2, 3, 99, 100]);
    }
}
