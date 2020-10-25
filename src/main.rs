mod neural_net;

use neural_net::Neuron;

fn main() {

    let weights = vec![1., 2., 3.];

    let neuron = Neuron::new(weights);

    println!("{:?}", neuron);

    let inputs = vec![1., 2., 1.];
    let output = neuron.forward(inputs);

    println!("{}", output);

}
