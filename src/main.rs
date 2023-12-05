mod bmd;
mod data;
mod distribution;
mod swords;

use std::collections::VecDeque;

use bmd::{Lookback12, RExp, BMD, SpikeDist};
use distribution::*;

use crate::bmd::{BlendedDist, WeightedCUDs};

fn main() -> std::io::Result<()> {

    swords::s12_locrot::go()?;
    // swords::s12_loc::go();

    return Ok(());
    /*
    let sumcud = AvgCUD {
        cuds: vec![
            CUD { a: 2.0, b: 3.0 },
            CUD { a: 10.0, b: 15.0 },
            CUD { a: 13.0, b: 18.0 },
        ],
    };

    let sumcud2 = AvgCUD {
        cuds: vec![
            CUD { a: -6.0, b: -2.0 },
            CUD { a: 0.0, b: 1.0 },
        ],
    };

    let wac = WeightedAvgCUD::from_weighted_avgcuds(&[&sumcud, &sumcud2], &[0.8, 0.2]);

    // for _ in 0..10_000 {
    //     println!("{}", wac.sample());
    // }

    let mut bmd: BMD<RExp, AvgCUD> = BMD { distributions: vec![] };

    bmd.distributions.push((RExp(101.0), sumcud));
    bmd.distributions.push((RExp(96.0), sumcud2));

    let point = bmd.interpolate::<WeightedAvgCUD>(RExp(100.0));
    for _ in 0..10_000 {
        println!("{}", point.sample());
    }

    return ();
    */

    let mut y = 10.0;
    let mut dy = 0.0;
    let ddy = -9.8;
    let frametime = 1.0 / 24.0;
    let mut frames = Vec::new();
    for frame in 0..120 {
        dy += ddy * frametime;
        y += dy * frametime;

        if y < 0.0 {
            y = 0.0;
            dy = 0.5 * f32::abs(dy);
        }
        // println!("{frame}\t{y}");
        frames.push(y)
    }

    {
        let mut bmd: BMD<RExp, AvgCUD> = BMD {
            distributions: vec![],
        };

        let delta = 0.1;

        for window in frames.windows(2) {
            bmd.distributions.push((
                RExp(window[0]),
                AvgCUD {
                    cuds: vec![CUD {
                        a: window[1] - delta,
                        b: window[1] + delta,
                    }],
                },
            ))
        }

        let mut last = 9.0;
        for i in 0..120 {
            // println!("{i}\t{last}");
            last = distribution::Sample::sample(&bmd.interpolate::<WeightedAvgCUD>(RExp(last)));
        }
    }

    {
        let delta = 0.2;

        let mut bmd12: BMD<Lookback12, AvgCUD> = BMD {
            distributions: vec![],
        };
        for window in frames.windows(13) {
            let mut buf = [0.0; 12];
            buf.clone_from_slice(&window[0..12]);
            bmd12.distributions.push((
                Lookback12(buf),
                AvgCUD {
                    cuds: vec![CUD {
                        a: window[12] - delta,
                        b: window[12] + delta,
                    }],
                },
            ))
        }

        dbg!(&bmd12);

        let mut last = frames[0..12].iter().map(|f| *f).collect::<VecDeque<_>>();
        let mut last = VecDeque::from([9.0; 12]);
        for i in 0..120 {
            // println!("{i}\t{}", last.back().unwrap());
            let mut buf = [0.0; 12];
            buf.copy_from_slice(last.make_contiguous());
            let new = Sample::sample(&bmd12.interpolate::<WeightedAvgCUD>(Lookback12(buf)));
            last.pop_front();
            last.push_back(new);
        }
    }

    {
        let delta = 0.2;

        let mut bmd12: BMD<Lookback12, CUD> = BMD {
            distributions: vec![],
        };

        for window in frames.windows(13) {
            let mut buf = [0.0; 12];
            buf.clone_from_slice(&window[0..12]);
            bmd12.distributions.push((
                Lookback12(buf),
                CUD {
                    a: window[12] - delta,
                    b: window[12] + delta,
                },
            ));
        }

        dbg!(&bmd12);

        let mut last = frames[0..12].iter().map(|f| *f).collect::<VecDeque<_>>();
        let mut last = VecDeque::from([9.0; 12]);
        for i in 0..120 {
            // println!("{i}\t{}", last.back().unwrap());
            let mut buf = [0.0; 12];
            buf.copy_from_slice(last.make_contiguous());
            let new = bmd12.interpolate::<WeightedCUDs>(Lookback12(buf)).sample();
            last.pop_front();
            last.push_back(new);
        }

        Ok(())
    }

}
