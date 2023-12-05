use rand::{
    distributions::WeightedIndex, prelude::Distribution, random, seq::SliceRandom, thread_rng,
};

use crate::distribution::{AvgCUD, Sample, WeightedAvgCUD, CUD, PDF};

// TODO: look into choose_weighted from random
// Blended Markov Distribution
#[derive(Debug)]
pub struct BMD<Pos: PositionState, Dist> {
    pub distributions: Vec<(Pos, Dist)>,
}

impl<'a, Pos: PositionState, Dist: 'a> BMD<Pos, Dist> {
    pub fn interpolate<T>(&'a self, eval_pos: Pos) -> T
    where
        T: BlendedDist<'a, &'a Dist> + 'a,
    {
        let weighted_dists = self
            .distributions
            .iter()
            .map(|(pos, dist)| (pos.similarity(&eval_pos), dist));

        T::from(weighted_dists)
    }
}

pub trait BlendedDist<'a, DistRef: 'a> {
    type OutputState;

    fn from<T>(weighted_dists: T) -> Self
    where
        T: IntoIterator<Item = (f32, DistRef)>;

    fn evalutate(&self, eval_pos: Self::OutputState) -> f32;

    fn sample(&self) -> Self::OutputState;
}

pub trait PositionState {
    fn similarity(&self, other: &Self) -> f32;
}

#[derive(Debug)]
pub struct Lookback12(pub [f32; 12]);

impl PositionState for Lookback12 {
    // just returning 1/distance for now
    fn similarity(&self, other: &Self) -> f32 {
        let distance = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(u, v)| (u - v) * (u - v))
            .sum::<f32>()
            .sqrt();

        const WIDENESS: f32 = 0.05;

        f32::exp(-distance / WIDENESS)
    }
}

pub struct RExp(pub f32);

impl PositionState for RExp {
    fn similarity(&self, other: &Self) -> f32 {
        let x = self.0 - other.0;

        // if x.abs() < 1.0

        const REXP_WIDENESS: f32 = 0.05;

        f32::exp(-(x * x) / REXP_WIDENESS)
    }
}

impl<'a> BlendedDist<'a, &'a AvgCUD> for WeightedAvgCUD<'a> {
    type OutputState = f32;

    fn from<T>(weighted_dists: T) -> Self
    where
        T: IntoIterator<Item = (f32, &'a AvgCUD)>,
    {
        let mut weights = Vec::new();
        let mut dists = Vec::new();
        // .unzip was being weird, so I did this
        for (w, d) in weighted_dists.into_iter() {
            weights.push(w);
            dists.push(d);
        }

        // normalize the weights, similarities between positionstates aren't garunteed to be normalized
        let total: f32 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= total);

        WeightedAvgCUD::from_weighted_avgcuds(&dists, &weights)
    }

    fn evalutate(&self, eval_pos: f32) -> f32 {
        <WeightedAvgCUD as PDF>::evaluate(&self, eval_pos)
    }

    fn sample(&self) -> f32 {
        <WeightedAvgCUD<'_> as Sample>::sample(&self)
    }
}

impl<'a> BlendedDist<'a, &'a CUD> for WeightedCUDs<'a> {
    type OutputState = f32;

    fn from<T>(weighted_dists: T) -> Self
    where
        T: IntoIterator<Item = (f32, &'a CUD)>,
    {
        let mut weights = Vec::new();
        let mut dists = Vec::new();
        // .unzip was being weird, so I did this
        for (w, d) in weighted_dists.into_iter() {
            weights.push(w);
            dists.push(d);
        }

        // NOTE: normalization is not necesary with the way I am doing sampling

        // normalize the weights, similarities between positionstates aren't garunteed to be normalized
        // let total: f32 = weights.iter().sum();
        // weights.iter_mut().for_each(|w| *w /= total);

        WeightedCUDs { weights, dists }
    }

    fn evalutate(&self, eval_pos: f32) -> f32 {
        // because we don't normalize weights, we have to calculate a sum here
        self.weights
            .iter()
            .zip(&self.dists)
            .map(|(w, cud)| cud.evaluate(eval_pos) * w)
            .sum::<f32>()
            / self.weights.iter().sum::<f32>()
    }

    fn sample(&self) -> f32 {
        let index = WeightedIndex::new(&self.weights).unwrap();
        let mut rng = thread_rng();

        self.dists[index.sample(&mut rng)].sample()
    }
}

pub struct WeightedCUDs<'a> {
    weights: Vec<f32>,
    dists: Vec<&'a CUD>,
}

#[derive(Debug)]
pub struct SpikeDist<O> {
    pub pos: O,
    pub side_len: f32,
}

impl<const N: usize> SpikeDist<[f32;N]>
{
    fn sample(&self) -> [f32;N] {
        let nudge = |inp: f32| inp + (self.side_len / 2.0) * (1.0 - random::<f32>() * 2.0);

        self.pos.map(nudge)
    }

}

pub struct WeightedSpikes<'a, Out> {
    dists: Vec<&'a SpikeDist<Out>>,
    weights: Vec<f32>,
}

impl<'a, const N: usize> BlendedDist<'a, &'a SpikeDist<[f32;N]>> for WeightedSpikes<'a, [f32;N]> {
    type OutputState = [f32;N];

    fn from<T>(weighted_dists: T) -> Self
    where
        T: IntoIterator<Item = (f32, &'a SpikeDist<[f32;N]>)>,
    {
        let mut weights = Vec::new();
        let mut dists = Vec::new();
        // .unzip was being weird, so I did this
        for (w, d) in weighted_dists.into_iter() {
            weights.push(w);
            dists.push(d);
        }
        // dbg!(&weights);

        WeightedSpikes { weights, dists }
    }

    fn evalutate(&self, eval_pos: [f32;N]) -> f32 {
        unimplemented!()
    }

    fn sample(&self) -> [f32;N] {
        let index = match WeightedIndex::new(&self.weights) {
            Ok(i) => i,
            Err(e) => panic!("We couldn't find anything similar!"),
        };
        let mut rng = thread_rng();

        self.dists[index.sample(&mut rng)].sample()
    }
}
