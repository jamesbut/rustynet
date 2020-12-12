use crate::neural_net::Layer;

pub fn gradient_descent(layer: &mut Layer) {

    let learning_rate = 0.01;
    for neuron in layer.neurons.iter_mut() {
        for i in 0..neuron.weights.len() {
            neuron.weights[i] -= learning_rate * neuron.gradients[i];
        }
    }

}
