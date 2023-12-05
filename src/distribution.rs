pub trait PDF {
    fn evaluate(&self, x: f32) -> f32;
}

#[derive(Debug)]
// Continuous Uniform Distribution
pub struct CUD {
    pub a: f32,
    pub b: f32,
}

impl CUD {
    pub fn len(&self) -> f32 {
        self.b - self.a
    }
}

impl PDF for CUD {
    fn evaluate(&self, x: f32) -> f32 {
        if self.a <= x && x <= self.b {
            1.0 / self.len()
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct AvgCUD {
    pub cuds: Vec<CUD>,
}

impl PDF for AvgCUD {
    fn evaluate(&self, x: f32) -> f32 {
        self.cuds.iter().map(|cud| cud.evaluate(x)).sum::<f32>() / self.cuds.len() as f32
    }
}

pub trait Sample {
    fn sample(&self) -> f32;
}

impl Sample for CUD {
    fn sample(&self) -> f32 {
        // return self.a + self.len() * rand::random::<f32>();

        //TODO: Clean this up. I'm trying to fake a smoother distrubution
        // We pick which chunk to sample as if each chunk were a proper chunk
        // But when we go to sample, we make sure to return things more towards the center than towards the corners
        //
        // Maybe we abstract this into something like rectangle-dist

        let center = (self.a + self.b) / 2.0;
        let half_width = self.len() / 2.0;

        let unit_spread = (rand::random::<f32>() - 0.5) * 2.0;

        center + half_width * (unit_spread * unit_spread * unit_spread)
    }
}

impl Sample for AvgCUD {
    fn sample(&self) -> f32 {
        let total = self.cuds.iter().map(CUD::len).sum::<f32>();

        let cutoff = rand::random::<f32>() * total;
        let mut bar = 0.0;
        for cud in &self.cuds {
            bar += cud.len();
            if bar > cutoff {
                return cud.sample();
            }
        }

        self.cuds.last().unwrap().sample()
    }
}

pub struct WeightedAvgCUD<'a> {
    weighted_cuds: Vec<(&'a CUD, f32)>,
}

impl<'a> WeightedAvgCUD<'a> {
    pub fn from_weighted_avgcuds(avg_cuds: &[&'a AvgCUD], weights: &[f32]) -> Self {
        assert_eq!(avg_cuds.len(), weights.len());

        let weighted_cuds = avg_cuds
            .iter()
            .zip(weights)
            .map(|(&avg_cud, w)| avg_cud.cuds.iter().map(|cud| (cud, *w)))
            .flatten()
            .collect();

        WeightedAvgCUD { weighted_cuds }
    }
}

impl Sample for WeightedAvgCUD<'_> {
    fn sample(&self) -> f32 {
        let total = self
            .weighted_cuds
            .iter()
            .map(|(cud, w)| cud.len() * w)
            .sum::<f32>();

        let cutoff = rand::random::<f32>() * total;
        let mut bar = 0.0;
        for (cud, w) in &self.weighted_cuds {
            bar += cud.len() * w;
            if bar > cutoff {
                return cud.sample();
            }
        }

        self.weighted_cuds.last().unwrap().0.sample()
    }
}

impl PDF for WeightedAvgCUD<'_> {
    fn evaluate(&self, x: f32) -> f32 {
        self.weighted_cuds
            .iter()
            .map(|(cud, w)| cud.evaluate(x) * w)
            .sum::<f32>()
    }
}
