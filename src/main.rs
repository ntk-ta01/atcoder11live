use rand::prelude::*;

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

const DIJ: [(usize, usize); 4] = [(0, 1), (1, 0), (0, !0), (!0, 0)];

type Output = Vec<(usize, usize)>;

const TIMELIMIT: f64 = 10.65;
fn main() {
    let mut timer = Timer::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let input = parse_input();
    let mut out = vec![];
    for _ in 0..input.n * input.n * 20 / 100 {
        let block = (rng.gen_range(0, input.n), rng.gen_range(0, input.n));
        if input.bs[block.0][block.1] == '#'
            || out.contains(&block)
            || block == (input.si, input.sj)
        {
            continue;
        }
        out.push(block);
    }
    annealing(&input, &mut out, &mut timer, &mut rng);

    write_output(&out);
}

fn annealing(
    input: &Input,
    output: &mut Output,
    timer: &mut Timer,
    rng: &mut rand_chacha::ChaCha20Rng,
) -> i64 {
    const T0: f64 = 10000.0;
    const T1: f64 = 100.0;
    let mut temp = T0;
    let mut prob;

    let mut count = 0;
    let now_state = State::new(input, output);
    let mut now_score = now_state.score;

    let mut best_score = now_score;
    let mut best_output = output.clone();
    loop {
        if count >= 100 {
            let passed = timer.get_time() / TIMELIMIT;
            if passed >= 1.0 {
                break;
            }
            temp = T0.powf(1.0 - passed) * T1.powf(passed);
            count = 0;
        }
        count += 1;

        let mut new_out = output.clone();
        // 近傍解生成。
        if new_out.is_empty() || rng.gen_bool(0.5) {
            let new_block = (rng.gen_range(0, input.n), rng.gen_range(0, input.n));
            if input.bs[new_block.0][new_block.1] == '#'
                || new_out.contains(&new_block)
                || new_block == (input.si, input.sj)
            {
                continue;
            }
            new_out.push(new_block);
        } else {
            let i = rng.gen_range(0, new_out.len());
            new_out.remove(i);
        }
        let new_state = State::new(input, output);
        let new_score = new_state.score;
        prob = f64::exp((new_score - now_score) as f64 / temp);
        if now_score < new_score || rng.gen_bool(prob) {
            // now_state = new_state;
            now_score = new_score;
            *output = new_out;
        }

        if best_score < now_score {
            best_score = now_score;
            best_output = output.clone();
        }
    }
    // eprintln!("{}", best_score);
    *output = best_output;
    best_score
}

#[derive(Clone, Debug)]
struct Input {
    n: usize,
    si: usize,
    sj: usize,
    bs: Vec<Vec<char>>,
}

fn parse_input() -> Input {
    use proconio::{input, marker::Chars};
    input! {
        n: usize,
        si: usize,
        sj: usize,
        bs: [Chars; n],
    }
    Input { n, si, sj, bs }
}

fn write_output(out: &Output) {
    println!("{}", out.len());
    for &(i, j) in out.iter() {
        println!("{} {}", i, j);
    }
}

#[allow(dead_code)]
fn compute_score(input: &Input, output: &Output) -> i64 {
    let s = State::new(input, output);
    s.score
}

struct State {
    score: i64,
    bs: Vec<Vec<char>>,
    pos: Vec<(usize, usize, usize)>,
}

impl State {
    fn new(input: &Input, out: &Output) -> State {
        let n = input.bs.len();
        let mut bs = input.bs.clone();
        for &(i, j) in out {
            if bs[i][j] != '.' {
                panic!("({}, {}) already has an obstacle.", i, j);
            } else if (i, j) == (input.si, input.sj) {
                panic!("You cannot place an obstacle in the initial position.");
            }
            bs[i][j] = 'o';
        }
        let mut visited = mat![false; n; n; 4];
        let mut dir = 0;
        let mut pi = input.si;
        let mut pj = input.sj;
        let mut pos = vec![(pi, pj, dir)];
        loop {
            if !visited[pi][pj][dir].setmax(true) {
                break;
            }
            let qi = pi + DIJ[dir].0;
            let qj = pj + DIJ[dir].1;
            if qi < n && qj < n && bs[qi][qj] == '.' {
                pi = qi;
                pj = qj;
                pos.push((pi, pj, dir));
            } else {
                dir = (dir + 1) % 4;
            }
        }
        let t = pos.len() - 1;
        let mut empty = 0;
        for i in 0..n {
            for j in 0..n {
                if input.bs[i][j] == '.' {
                    empty += 1;
                }
            }
        }
        let score = (1e6 * t as f64 / (4.0 * empty as f64)).round() as i64;
        State { score, bs, pos }
    }
}

trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

fn get_time() -> f64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9
}

struct Timer {
    start_time: f64,
}

impl Timer {
    fn new() -> Timer {
        Timer {
            start_time: get_time(),
        }
    }

    fn get_time(&self) -> f64 {
        get_time() - self.start_time
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.start_time = 0.0;
    }
}
