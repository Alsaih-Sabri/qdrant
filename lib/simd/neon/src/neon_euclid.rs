use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
#[allow(unused)]
pub unsafe fn neon_euclid_similarity_bytes(v1: &[u8], v2: &[u8]) -> f32 {
    debug_assert!(v1.len() == v2.len());
    let mut ptr1: *const u8 = v1.as_ptr();
    let mut ptr2: *const u8 = v2.as_ptr();

    let mut mul1 = vdupq_n_u32(0);
    let mut mul2 = vdupq_n_u32(0);
    let len = v1.len();
    for _ in 0..len / 16 {
        let p1 = vld1q_u8(ptr1);
        let p2 = vld1q_u8(ptr2);
        ptr1 = ptr1.add(16);
        ptr2 = ptr2.add(16);

        let abs_diff = vabdq_u8(p1, p2);
        let abs_diff_low = vget_low_u8(abs_diff);
        let abs_diff_high = vget_high_u8(abs_diff);

        let mul_low = vmull_u8(abs_diff_low, abs_diff_low);
        let mul_high = vmull_u8(abs_diff_high, abs_diff_high);

        mul1 = vpadalq_u16(mul1, mul_low);
        mul2 = vpadalq_u16(mul2, mul_high);
    }
    let mut score = vaddvq_u32(vaddq_u32(mul1, mul2)) as f32;

    let mut remainder = len % 16;
    if remainder != 0 {
        let mut remainder_score = 0;
        for _ in 0..remainder {
            let v1 = *ptr1 as i32;
            let v2 = *ptr2 as i32;
            ptr1 = ptr1.add(1);
            ptr2 = ptr2.add(1);
            let diff = v1 - v2;
            remainder_score += diff * diff;
        }
        score += remainder_score as f32;
    }

    -score
}

#[cfg(test)]
mod tests {
    use std::arch::is_aarch64_feature_detected;

    use super::*;

    fn euclid_similarity_bytes(v1: &[u8], v2: &[u8]) -> f32 {
        -v1.iter()
            .zip(v2)
            .map(|(a, b)| {
                let diff = *a as i32 - *b as i32;
                diff * diff
            })
            .sum::<i32>() as f32
    }

    #[test]
    fn test_spaces_sse() {
        if is_aarch64_feature_detected!("neon") {
            let v1: Vec<u8> = vec![
                255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3,
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17,
            ];
            let v2: Vec<u8> = vec![
                255, 255, 0, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241,
                240, 239, 238, 255, 255, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245,
                244, 243, 242, 241, 240, 239, 238, 255, 255, 255, 254, 253, 252, 251, 250, 249,
                248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 255, 255, 255, 254, 253,
                252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 255,
                255, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241,
                240, 239, 238,
            ];

            let dot_simd = unsafe { neon_euclid_similarity_bytes(&v1, &v2) };
            let dot = euclid_similarity_bytes(&v1, &v2);
            assert_eq!(dot_simd, dot);
        } else {
            println!("avx test skipped");
        }
    }
}
