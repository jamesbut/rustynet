#[derive(Debug)]
pub struct Neuron {
    
    pub weights: Vec<f64>,

    pub gradients: Vec<f64>,
    
    //Keep track of the inputs on the forward pass to be used in the 
    //gradient calculation
    inputs: Vec<f64>,

}

impl Neuron {

    pub fn new(weights: Vec<f64>) -> Self {
        let gradients = vec![0.; weights.len()];
        let inputs = vec![0.; weights.len()];
        Neuron {
            weights,
            gradients,
            inputs,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> f64 {

        let mut sum = 0.;
        for i in 0..inputs.len() {
            sum += inputs[i] * self.weights[i];
        }

        self.inputs = inputs.to_vec();

        sum

    }

    //Computes the gradients
    pub fn backward(&mut self, output_grad: f64) {
        for i in 0..self.gradients.len() {
            self.gradients[i] = self.inputs[i] * output_grad;
        }
    }

    pub fn zero_grad(&mut self) {
        for grad in &mut self.gradients {
            *grad = 0.;
        }
    }

}
