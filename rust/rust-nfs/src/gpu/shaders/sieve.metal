#include <metal_stdlib>
using namespace metal;

struct BucketUpdate {
    ushort pos;
    uchar logp;
} __attribute__((packed));

kernel void apply_bucket_updates(
    device atomic_uint *sieve_array [[buffer(0)]],
    device const BucketUpdate *updates [[buffer(1)]],
    device const uint *sieve_len [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint total_updates [[threads_per_grid]]
) {
    if (id >= total_updates) {
        return;
    }

    BucketUpdate update = updates[id];

    if ((uint)update.pos >= *sieve_len) {
        return;
    }

    uint word_idx = update.pos / 4;

    // Prevent out of bounds reads/writes to the 32-bit chunk representation.
    // If word_idx * 4 >= *sieve_len, we could be indexing beyond the array allocation.
    // In practice, we require the host to pad the sieve array size to a multiple of 4 bytes.
    if (word_idx * 4 >= *sieve_len) {
        return;
    }

    uint byte_offset = update.pos % 4;
    uint shift = byte_offset * 8;

    uint expected = atomic_load_explicit(&sieve_array[word_idx], memory_order_relaxed);
    uint desired;
    bool success = false;

    while (!success) {
        uint current_byte = (expected >> shift) & 0xFF;
        uint new_byte;
        if (current_byte > update.logp) {
            new_byte = current_byte - update.logp;
        } else {
            new_byte = 0; // Saturated subtraction
        }

        uint mask = ~(0xFFu << shift);
        desired = (expected & mask) | (new_byte << shift);

        success = atomic_compare_exchange_weak_explicit(
            &sieve_array[word_idx],
            &expected,
            desired,
            memory_order_relaxed,
            memory_order_relaxed
        );
    }
}
