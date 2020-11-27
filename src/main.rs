mod neural_net;

use neural_net::Neuron;

fn mean_squared_error(output: f64, target: f64) -> f64 {
    (output - target).powi(2)
}

fn output_gradient(output: f64, target: f64) -> f64 {
    2. * (output - target)
}

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

    println!("{:?}", neuron);

    let inputs = vec![2.];

    let num_epochs = 100;
    for i in 0..num_epochs {
        let output = neuron.forward(&inputs);

        //println!("Output: {}", output);
        let target = 1.;
        let loss = mean_squared_error(output, target);

        println!("Loss: {}", loss);

        neuron.backward(output_gradient(output, target));

        //println!("{:?}", neuron);

        gradient_descent(&mut neuron);

        println!("{:?}", neuron);
        neuron.zero_grad();

    }
}
