use crate::neural_net::Network;

pub fn gradient_descent(network: &mut Network) {

    let learning_rate = 0.01;
    for layer in network.layers.iter_mut() {
        for neuron in layer.neurons.iter_mut() {
            for i in 0..neuron.weights.len() {
                neuron.weights[i] -= learning_rate * neuron.gradients[i];
            }
        }
    }

}
