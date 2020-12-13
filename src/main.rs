mod neural_net;
mod loss;
mod activation_functions;
mod optimiser;

use neural_net::Layer;
use loss::MSE;
use optimiser::gradient_descent;


fn main() {

    let mut loss_function = MSE::new();

    let num_neurons = 2;
    let inputs_per_neuron = 2;

    let mut layer = Layer::new(num_neurons, inputs_per_neuron);

    println!("{:?}", layer);

    let inputs = vec![2., 3.];
    let targets = vec![1., 0.];

    let num_epochs = 10000;
    for i in 0..num_epochs {
        println!("---------------\nEpoch: {}", i);

        println!("Inputs: {:?}", inputs);
        let outputs = layer.forward(&inputs);

        println!("Output: {:?}", outputs);
        let loss = loss_function.forward(&outputs, &targets);

        println!("Targets: {:?}", targets);
        println!("Loss: {:?}", loss);

        loss_function.backward();
        layer.backward(&loss_function.gradients);

        gradient_descent(&mut layer);

        println!("{:?}", layer);
        layer.zero_grad();

    }

}
