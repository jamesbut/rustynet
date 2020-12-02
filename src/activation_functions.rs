#[derive(Debug)]
pub struct Sigmoid {

    //Keep track of input for gradient calculation
    input: f64,

    pub gradient: f64,

}

impl Sigmoid {

    pub fn new() -> Self {
        Sigmoid {
            input: 0.,
            gradient: 0.
        }
    }

    pub fn forward(&mut self, x: f64) -> f64 {
        self.input = x;
        return self.sig(x);
    }
    
    pub fn backward(&mut self, output_grad: f64) {
        self.gradient = output_grad * self.sig(self.input) * (1. - self.sig(self.input));
    }

    fn sig(&self, x: f64) -> f64 {
        return x.exp() / (x.exp() + 1.);
    }

}
