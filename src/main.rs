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
    let num_hidden_layers = 0;
    let neurons_per_hidden_layer = 2;

    let mut network = Network::new(num_inputs, num_outputs, num_hidden_layers,
                                   neurons_per_hidden_layer);

    println!("{:?}", network);

    let inputs = vec![2., 3.];
    let targets = vec![0.6, 0.4];

    let num_epochs = 10000;
    for i in 0..num_epochs {
        println!("---------------\nEpoch: {}", i);

        println!("Inputs: {:?}", inputs);
        let outputs = network.forward(&inputs);

        println!("Output: {:?}", outputs);
        let loss = loss_function.forward(&outputs, &targets);

        println!("Targets: {:?}", targets);
        println!("Loss: {:?}", loss);

        loss_function.backward();
        network.backward(&loss_function.gradients);

        gradient_descent(&mut network);

        println!("{:?}", network);
        network.zero_grad();

    }

}
