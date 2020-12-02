use crate::activation_functions::Sigmoid;

#[derive(Debug)]
pub struct Neuron {
    
    pub weights: Vec<f64>,

    pub gradients: Vec<f64>,
    
    //Keep track of the inputs on the forward pass to be used in the 
    //gradient calculation
    inputs: Vec<f64>,

    sigmoid: Sigmoid,

}

impl Neuron {

    pub fn new(weights: Vec<f64>) -> Self {
        let gradients = vec![0.; weights.len()];
        let inputs = vec![0.; weights.len()];
        let sigmoid = Sigmoid::new();
        Neuron {
            weights,
            gradients,
            inputs,
            sigmoid,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> f64 {

        let mut activation = 0.;
        for i in 0..inputs.len() {
            activation += inputs[i] * self.weights[i];
        }

        let output = self.sigmoid.forward(activation);

        self.inputs = inputs.to_vec();

        output

    }

    //Computes the gradients
    pub fn backward(&mut self, output_grad: f64) {
        self.sigmoid.backward(output_grad);
        for i in 0..self.gradients.len() {
            self.gradients[i] = self.inputs[i] * self.sigmoid.gradient;
        }
    }

    pub fn zero_grad(&mut self) {
        for grad in &mut self.gradients {
            *grad = 0.;
        }
    }

}
