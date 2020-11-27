#[derive(Debug)]
pub struct MSE {

    //Store these values on the forward pass so one can just call backward
    //and have all the required information
    outputs: Vec<f64>,
    targets: Vec<f64>,

    pub gradients: Vec<f64>,

}

impl MSE {

    pub fn new() -> Self {
        let outputs = Vec::new();
        let targets = Vec::new();
        let gradients = Vec::new();
        MSE {
            outputs,
            targets,
            gradients,
        }
    }

    pub fn forward(&mut self, outputs: &Vec<f64>, targets: &Vec<f64>) -> Vec<f64> {
        self.outputs = outputs.to_vec();
        self.targets = targets.to_vec();
        let mut losses = vec![0.; outputs.len()];
        for i in 0..losses.len() {
            losses[i] = (outputs[i] - targets[i]).powi(2);
        }
        losses
    }

    pub fn backward(&mut self) {
        self.gradients = vec![0.; self.outputs.len()];
        for i in 0..self.outputs.len() {
            self.gradients[i] = 2. * (self.outputs[i] - self.targets[i])
        }
    }

}
