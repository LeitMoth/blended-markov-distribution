use std::collections::VecDeque;

use crate::{bmd::{PositionState, BMD, SpikeDist, WeightedSpikes, BlendedDist}, data};

const OUT: usize = 3;
const LOOKBACK: usize = 1;

impl PositionState for [f32;OUT*LOOKBACK] {
    fn similarity(&self, other: &Self) -> f32 {
        let distance = self
            .iter()
            .zip(other.iter())
            .map(|(u, v)| (u - v) * (u - v))
            .sum::<f32>()
            .sqrt();

        const WIDENESS: f32 = 0.05;

        f32::exp(-distance / WIDENESS)
    }
}

pub fn go() {
    let delta = 0.1;

    let mut sword_bmd: BMD<[f32;OUT*LOOKBACK], SpikeDist<[f32;OUT]>> = BMD {
        distributions: vec![],
    };

    let dif_data: Vec<[f32;OUT]> = data::sword_6().windows(2).map(|win| {
        let mut buf = [0.0; OUT];

        for i in 0..buf.len() {
            buf[i] = win[1][i] - win[0][i];
        }
        buf
    }).collect();

    let data: Vec<[f32; OUT]> = data::sword_6().iter().map(|l| [l[0], l[1], l[2]]).collect();

    let mut buf = [0.0; OUT*LOOKBACK];

    for window in data.windows(LOOKBACK + 1) { // replace with dif_ for dif
        for (i, chunk) in buf.chunks_mut(OUT).enumerate() {
            chunk.clone_from_slice(&window[i]);
        }
        sword_bmd.distributions.push((
            buf,
            SpikeDist {
                pos: window[LOOKBACK],
                side_len: delta
            }
        ));
    }

    dbg!(&sword_bmd);

    let mut current = [0.0; OUT];

    let mut last = VecDeque::from_iter(data.iter().take(LOOKBACK).map(|l| l.clone()));

    for i in 0..150 {
        let last_buf = last.back().unwrap();
        for i in 0..current.len() {
            current[i] += last_buf[i];
        }
        println!("{:?}", last_buf);
        let mut buf = [0.0; OUT*LOOKBACK];

        for (j, chunk) in buf.chunks_mut(OUT).enumerate() {
            chunk.clone_from_slice(&last[j]);
        }
        let new = sword_bmd.interpolate::<WeightedSpikes<[f32;OUT]>>(buf).sample();
        last.pop_front();
        last.push_back(new);
    }
}
