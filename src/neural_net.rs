#[derive(Debug)]
pub struct Neuron {
    
    weights: Vec<f64>,

}

impl Neuron {

    pub fn new(weights: Vec<f64>) -> Self {
        Neuron {
            weights,
        }
    }

    pub fn forward(&self, inputs: Vec<f64>) -> f64 {

        let mut sum = 0.;
        for i in 0..inputs.len() {
            sum += inputs[i] * self.weights[i];
        }

        sum

    }

}
