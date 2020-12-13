mod neural_net;
mod loss;
mod activation_functions;
mod optimiser;

use neural_net::Network;
use loss::MSE;
use optimiser::gradient_descent;


fn main() {

    let mut loss_function = MSE::new();

    let num_inputs = 2;
    let num_outputs = 2;
    let num_hidden_layers = 1;
    let neurons_per_hidden_layer = 2;

    //let mut layer = Layer::new(num_neurons, inputs_per_neuron);
    let mut network = Network::new(num_inputs, num_outputs, num_hidden_layers,
                                   neurons_per_hidden_layer);

    println!("{:?}", network);

    let inputs = vec![2., 3.];
    let targets = vec![1., 0.];

    let num_epochs = 1;
    for i in 0..num_epochs {
        println!("---------------\nEpoch: {}", i);

        println!("Inputs: {:?}", inputs);
        let outputs = network.forward(&inputs);

        println!("Output: {:?}", outputs);
        /*
        let loss = loss_function.forward(&outputs, &targets);

        println!("Targets: {:?}", targets);
        println!("Loss: {:?}", loss);

        loss_function.backward();
        layer.backward(&loss_function.gradients);

        gradient_descent(&mut layer);

        println!("{:?}", layer);
        layer.zero_grad();
        */

    }

}
