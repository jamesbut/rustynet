use crate::activation_functions::Sigmoid;
use rand::distributions::{Normal, Distribution};

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

    //TODO: Add option for bias
    pub fn new_w_weight(weights: Vec<f64>) -> Self {
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

    //TODO: Add option for bias
    pub fn new(num_inputs: usize) -> Self {
        let gradients = vec![0.; num_inputs+1];
        let inputs = vec![0.; num_inputs];
        let sigmoid = Sigmoid::new();
        let weights = Neuron::initialise_weights(num_inputs);
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
        //Bias
        *self.gradients.last_mut().unwrap() = self.sigmoid.gradient;
    }

    pub fn zero_grad(&mut self) {
        for grad in &mut self.gradients {
            *grad = 0.;
        }
    }

    fn initialise_weights(num_inputs: usize) -> Vec<f64> {
        
        // +1 for bias
        let mut init_weights = vec![0.; num_inputs+1];

        //Initialise weights from gaussian
        let normal = Normal::new(0., 1.);

        for weight in init_weights.iter_mut() {
            *weight = 1.;
            //*weight = normal.sample(&mut rand::thread_rng()); 
        }

        init_weights

    }

}

#[derive(Debug)]
pub struct Layer {

    pub neurons: Vec<Neuron>

}

impl Layer {

    pub fn new(num_neurons: usize, inputs_per_neuron: usize) -> Self {
        
        let mut neurons : Vec<Neuron> = Vec::with_capacity(num_neurons);
        for _ in 0..num_neurons {
            neurons.push(Neuron::new(inputs_per_neuron)); 
        }

        Layer {
            neurons,
        }

    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {

        let mut outputs = Vec::with_capacity(self.neurons.len());
        for neuron in self.neurons.iter_mut() {
            outputs.push(neuron.forward(inputs));
        }
        outputs
         
    }

    pub fn backward(&mut self, output_grads: &Vec<f64>) {
        for i in 0..output_grads.len() {
            self.neurons[i].backward(output_grads[i]);
        }
    }

    pub fn zero_grad(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.zero_grad();
        }
    }

}

#[derive(Debug)]
pub struct Network {

    pub layers: Vec<Layer>

}

impl Network {

    pub fn new(num_inputs: usize, num_outputs: usize, 
               num_hidden_layers: usize, neurons_per_hidden_layer: usize) -> Self {
        
        let mut layers : Vec<Layer> = Vec::with_capacity(num_hidden_layers+1);

        if num_hidden_layers == 0 {
            layers.push(Layer::new(num_outputs, num_inputs));
        } else {

            layers.push(Layer::new(neurons_per_hidden_layer, num_inputs));

            for _ in 0..num_hidden_layers-1 {
                layers.push(Layer::new(neurons_per_hidden_layer, neurons_per_hidden_layer));
            }

            layers.push(Layer::new(num_outputs, neurons_per_hidden_layer));

        }

        Network {
            layers
        }

    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        
        let mut curr_inputs = inputs.to_owned();
        for layer in self.layers.iter_mut() {
            curr_inputs = layer.forward(&curr_inputs);
        }

        curr_inputs

    }

    pub fn backward(&mut self, loss_grads: &Vec<f64>) {

    }

}
