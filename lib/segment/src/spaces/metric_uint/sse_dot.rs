use std::arch::x86_64::*;

use common::types::ScoreType;

use crate::data_types::vectors::VectorElementTypeByte;
use crate::spaces::simple_sse::hsum128_ps_sse;

#[target_feature(enable = "sse")]
#[allow(unused)]
pub unsafe fn sse_dot_similarity_bytes(
    v1: &[VectorElementTypeByte],
    v2: &[VectorElementTypeByte],
) -> ScoreType {
    debug_assert!(v1.len() == v2.len());
    let mut ptr1: *const VectorElementTypeByte = v1.as_ptr();
    let mut ptr2: *const VectorElementTypeByte = v2.as_ptr();

    // sum accumulator for 4x32 bit integers
    let mut acc = _mm_setzero_si128();
    // mask to take only lower 8 bits from 16 bits
    let mask_epu16_epu8 = _mm_set1_epi16(0xFF);
    // mask to take only lower 16 bits from 32 bits
    let mask_epu32_epu16 = _mm_set1_epi32(0xFFFF);
    let len = v1.len();
    for _ in 0..len / 16 {
        // load 16 bytes
        let p1 = _mm_loadu_si128(ptr1 as *const __m128i);
        let p2 = _mm_loadu_si128(ptr2 as *const __m128i);
        ptr1 = ptr1.add(16);
        ptr2 = ptr2.add(16);

        // take from lane p1 and p2 parts (using bitwise AND):
        // p1 = [byte0, byte1, byte2, byte3, ..] -> [0, byte1, 0, byte3, ..]
        // p2 = [byte0, byte1, byte2, byte3, ..] -> [0, byte1, 0, byte3, ..]
        // and calculate 16bit multiplication with taking lower 16 bits
        // wa can use signed multiplication because sign bit is always 0
        let mul16 = _mm_mullo_epi16(
            _mm_and_si128(p1, mask_epu16_epu8),
            _mm_and_si128(p2, mask_epu16_epu8),
        );

        acc = _mm_add_epi32(acc, _mm_and_si128(mul16, mask_epu32_epu16));
        let mul16 = _mm_bsrli_si128(mul16, 2);
        acc = _mm_add_epi32(acc, _mm_and_si128(mul16, mask_epu32_epu16));

        // shift right by 1 byte for p1 and p2 and repeat previous steps
        let p1 = _mm_bsrli_si128(p1, 1);
        let p2 = _mm_bsrli_si128(p2, 1);

        let mul16 = _mm_mullo_epi16(
            _mm_and_si128(p1, mask_epu16_epu8),
            _mm_and_si128(p2, mask_epu16_epu8),
        );

        acc = _mm_add_epi32(acc, _mm_and_si128(mul16, mask_epu32_epu16));
        let mul16 = _mm_bsrli_si128(mul16, 2);
        acc = _mm_add_epi32(acc, _mm_and_si128(mul16, mask_epu32_epu16));
    }

    let mul_ps = _mm_cvtepi32_ps(acc);
    let score = hsum128_ps_sse(mul_ps);

    let mut remainder = 0;
    for _ in 0..len % 16 {
        let v1 = *ptr1;
        let v2 = *ptr2;
        ptr1 = ptr1.add(1);
        ptr2 = ptr2.add(1);
        remainder += (v1 as i32) * (v2 as i32);
    }

    if remainder != 0 {
        score + remainder as f32
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::metric::Metric;
    use crate::spaces::simple::*;

    #[test]
    fn test_spaces_sse() {
        if is_x86_feature_detected!("sse") {
            let v1: Vec<VectorElementTypeByte> = vec![
                255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3,
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17,
            ];
            let v2: Vec<VectorElementTypeByte> = vec![
                255, 255, 0, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241,
                240, 239, 238, 255, 255, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245,
                244, 243, 242, 241, 240, 239, 238, 255, 255, 255, 254, 253, 252, 251, 250, 249,
                248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 255, 255, 255, 254, 253,
                252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 255,
                255, 255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241,
                240, 239, 238,
            ];

            let dot_simd = unsafe { sse_dot_similarity_bytes(&v1, &v2) };
            let dot = <DotProductMetric as Metric<VectorElementTypeByte>>::similarity(&v1, &v2);
            assert_eq!(dot_simd, dot);
        } else {
            println!("avx test skipped");
        }
    }
}
