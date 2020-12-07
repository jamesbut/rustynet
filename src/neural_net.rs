use crate::activation_functions::Sigmoid;

#[derive(Debug)]
pub struct Neuron {
    
    //Bias is the last value in the vector
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
        let inputs = vec![0.; weights.len()-1];
        let sigmoid = Sigmoid::new();
        Neuron {
            weights,
            gradients,
            inputs,
            sigmoid,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> f64 {

        if inputs.len() != self.inputs.len() {
            eprintln!("The number of inputs to the network is not equal to the number of \
                       weights minus one!\nNum inputs: {}, Num weights: {}", inputs.len(),
                       self.gradients.len());
            std::process::exit(0);
        }

        let mut activation = 0.;
        for i in 0..inputs.len() {
            activation += inputs[i] * self.weights[i];
        }

        //Bias
        activation += self.weights.last().copied().unwrap();

        let output = self.sigmoid.forward(activation);

        self.inputs = inputs.to_vec();

        output

    }

    //Computes the gradients
    pub fn backward(&mut self, output_grad: f64) {
        self.sigmoid.backward(output_grad);
        for i in 0..self.inputs.len() {
            self.gradients[i] = self.inputs[i] * self.sigmoid.gradient;
        }
        *self.gradients.last_mut().unwrap() = self.sigmoid.gradient;
    }

    pub fn zero_grad(&mut self) {
        for grad in &mut self.gradients {
            *grad = 0.;
        }
    }

}
