mod neural_net;
mod loss;
mod activation_functions;

//use neural_net::Neuron;
use neural_net::Layer;
use loss::MSE;

/*
fn gradient_descent(neuron: &mut Neuron) {

    let learning_rate = 0.01;
    for i in 0..neuron.weights.len() {
        neuron.weights[i] -= learning_rate * neuron.gradients[i];
    }

}
*/

fn main() {

    let mut loss_function = MSE::new();

    //let weights = vec![-2., 1.];
    let num_neurons = 2;
    let inputs_per_neuron = 2;

    let mut layer = Layer::new(num_neurons, inputs_per_neuron);

    //println!("{:?}", neuron);
    println!("{:?}", layer);

    let inputs = vec![2., 3.];

    let outputs = layer.forward(&inputs);

    println!("{:?}", outputs);

    std::process::exit(0);

    /*
    let num_epochs = 1;
    for i in 0..num_epochs {
        println!("Epoch: {}", i);

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
    */
}
