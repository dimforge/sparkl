//! See https://stackoverflow.com/questions/50210379/3d-morton-encoding-using-bit-interleaving-conventional-vs-bmi2-instruction-set

fn spread(mut w: u64) -> u64 {
    w &= 0x00000000001fffff;
    w = (w | w << 32) & 0x001f00000000ffff;
    w = (w | w << 16) & 0x001f0000ff0000ff;
    w = (w | w << 8) & 0x010f00f00f00f00f;
    w = (w | w << 4) & 0x10c30c30c30c30c3;
    w = (w | w << 2) & 0x1249249249249249;
    w
}

pub fn morton_encode3(x: u32, y: u32, z: u32) -> u64 {
    spread(x as u64) | (spread(y as u64) << 1) | (spread(z as u64) << 2)
}

///////////////// For Decoding //////////////////////

fn compact(mut w: u64) -> u32 {
    w &= 0x1249249249249249;
    w = (w ^ (w >> 2)) & 0x30c30c30c30c30c3;
    w = (w ^ (w >> 4)) & 0xf00f00f00f00f00f;
    w = (w ^ (w >> 8)) & 0x00ff0000ff0000ff;
    w = (w ^ (w >> 16)) & 0x00ff00000000ffff;
    w = (w ^ (w >> 32)) & 0x00000000001fffff;
    w as u32
}

pub fn morton_decode3(code: u64) -> [u32; 3] {
    [compact(code), compact(code >> 1), compact(code >> 2)]
}
