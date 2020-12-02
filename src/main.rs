mod neural_net;
mod loss;
mod activation_functions;

use neural_net::Neuron;
use loss::MSE;

fn gradient_descent(neuron: &mut Neuron) {

    let learning_rate = 0.01;
    for i in 0..neuron.weights.len() {
        neuron.weights[i] -= learning_rate * neuron.gradients[i];
    }

}

fn main() {

    //let weights = vec![1., 2., 3.];
    let weights = vec![-2.];

    let mut neuron = Neuron::new(weights);
    let mut loss_function = MSE::new();

    println!("{:?}", neuron);

    let inputs = vec![2.];

    let num_epochs = 10000;
    for _i in 0..num_epochs {
        let output = neuron.forward(&inputs);

        println!("Output: {}", output);
        let outputs = vec![output];
        let targets = vec![1.];
        let loss = loss_function.forward(&outputs, &targets);

        println!("Loss: {:?}", loss);

        loss_function.backward();
        neuron.backward(loss_function.gradients[0]);

        //println!("{:?}", neuron);

        gradient_descent(&mut neuron);

        println!("{:?}", neuron);
        neuron.zero_grad();

    }
}
