use std::{collections::VecDeque, io::BufWriter, fs::File, path::Path};

use crate::{bmd::{PositionState, BMD, SpikeDist, WeightedSpikes, BlendedDist}, data};

struct LB12Dot(pub [f32;72]);
struct LB12Temporal(pub [f32;72]);

impl PositionState for [f32;72] {
    fn similarity(&self, other: &Self) -> f32 {
        let distance = self
            .iter()
            .zip(other.iter())
            .enumerate()
            .map(|(i, (u, v))| (u - v) * (u - v) * (i as f32 + 1.0))
            .sum::<f32>()
            .sqrt();

        const WIDENESS: f32 = 0.1;

        // dbg!(distance);

        f32::exp(-distance / WIDENESS)
    }
}

impl PositionState for LB12Dot {
    fn similarity(&self, other: &Self) -> f32 {
        let distance = self.0
            .iter()
            .zip(other.0.iter())
            .enumerate()
            .map(|(i, (u, v))| (u - v) * (u - v) * (i as f32 + 1.0))
            .sum::<f32>()
            .sqrt();

        const WIDENESS: f32 = 0.1;

        // dbg!(distance);

        f32::exp(-distance / WIDENESS)
    }
}

// impl PositionState for LB12Temporal {
//     fn similarity(&self, other: &Self) -> f32 {
//         let distance = self.0
//             .iter()
//             .zip(other.0.iter())
//             .enumerate()
//             .map(|(i, (u, v))| (u - v) * (u - v) * (i as f32 + 1.0))
//             .sum::<f32>()
//             .sqrt();

//         const WIDENESS: f32 = 0.1;

//         // dbg!(distance);

//         f32::exp(-distance / WIDENESS)
//     }
// }

pub fn go() -> std::io::Result<()> {
    let delta = 0.0;

    let mut sword_bmd: BMD<[f32;6*12], SpikeDist<[f32;6]>> = BMD {
        distributions: vec![],
    };

    let dif_data: Vec<[f32;6]> = data::sword_6().windows(2).map(|win| {
        let mut buf = [0.0; 6];
        for i in 0..6 {
            buf[i] = win[1][i] - win[0][i];
        }
        buf
    }).collect();

    // let dif_data = data::sword_6();

    let mut buf = [0.0; 6*12];

    for window in dif_data.windows(13) {
        for (i, chunk) in buf.chunks_mut(6).enumerate() {
            chunk.clone_from_slice(&window[i]);
        }
        sword_bmd.distributions.push((
            buf,
            SpikeDist {
                pos: window[12],
                side_len: delta
            }
        ));
    }

    let mut current = data::sword_6()[12];

    let mut last = VecDeque::from_iter(std::iter::repeat(
[-1.5459953546524048, -0.3006895184516907, 4.337007522583008, 0.3151423931121826, 0.016330672428011894, -1.4718228578567505]
    ).take(12));

    let mut last = VecDeque::from_iter(dif_data.iter().take(12).map(|l| l.clone()));


    let f = File::create("./12s_locrot_dot.pos")?;

    for i in 0..150 {
        let last_buf = last.back().unwrap();
        for i in 0..6 {
            current[i] += last_buf[i];
        }
        println!("{:?}", current);
        let mut buf = [0.0; 72];

        for (j, chunk) in buf.chunks_mut(6).enumerate() {
            chunk.clone_from_slice(&last[j]);
        }
        let new = sword_bmd.interpolate::<WeightedSpikes<[f32;6]>>(buf).sample();
        last.pop_front();
        last.push_back(new);
    }

    return Ok(())
}
