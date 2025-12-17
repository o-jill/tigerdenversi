use super::*;

const INPUTSIZE :i64 = weight::N_INPUT as i64;
const HIDDENSIZE : i64 = weight::N_HIDDEN as i64;
const HIDDENSIZE2 : i64 = weight::N_HIDDEN2 as i64;

pub fn net(vs : &nn::Path) -> impl Module {
    let relu = true;
    // let relu = false;  // sigmoid
    if relu {
        // println!("activation function: RELU");
        nn::seq()
            .add(
                nn::linear(vs / "layer1",
                INPUTSIZE, HIDDENSIZE, Default::default()))
                // INPUTSIZE, HIDDENSIZE, LinearConfig{ws_init:tch::nn::Init::set(self, tensor)}))
            .add_fn(|xs| xs.relu())
            .add(
                nn::linear(vs / "layer2",
                 HIDDENSIZE, HIDDENSIZE2, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "layer3", HIDDENSIZE2, 1, Default::default()))
    } else {
        // println!("activation function: Sigmoid");
        nn::seq()
            .add(
                nn::linear(vs / "layer1",
                INPUTSIZE, HIDDENSIZE, Default::default()))
                // INPUTSIZE, HIDDENSIZE, LinearConfig{ws_init:tch::nn::Init::set(self, tensor)}))
            .add_fn(|xs| xs.sigmoid())
            .add(
                nn::linear(vs / "layer2",
                 HIDDENSIZE, HIDDENSIZE2, Default::default()))
            .add_fn(|xs| xs.sigmoid())
            .add(nn::linear(vs / "layer3", HIDDENSIZE2, 1, Default::default()))
    }
}

pub fn loadtensor(vs : &mut VarStore, key : &str, src : &Tensor) {
    let mut val = vs.variables_.lock();
    let s = val.as_mut().unwrap().named_variables.get_mut(key).unwrap();
    s.set_data(src);
}

// load from `fname` into `vs`.
//
// .safetensor and ruversi weight format are available.
pub fn load(vs : &mut VarStore, weights_org : &weight::Weight, progress : usize)
        -> Result<(), String> {
    const INPSIZE : usize = INPUTSIZE as usize;
    const HIDSIZE : usize = HIDDENSIZE as usize;
    let wban = weights_org.wban(progress);
    let wtbn = weights_org.wteban(progress);
    let wfs = weights_org.wfixedstones(progress);
    let wdc = weights_org.wibias(progress);
    let whdn = weights_org.wlayer1(progress);
    let wdc2 = weights_org.wl1bias(progress);
    let whdn2 = weights_org.wlayer2(progress);
    let wdc3 = weights_org.wl2bias(progress);

    // layer1.weight
    let mut weights = [0.0f32 ; INPSIZE * HIDSIZE];
    for i in 0..HIDDENSIZE as usize {
        let wsz = bitboard::CELL_2D * 2;
        weights[i * INPSIZE..i * INPSIZE + wsz].copy_from_slice(
            &wban[i * wsz .. (i + 1) * wsz]);
        weights[i * INPSIZE + wsz] = wtbn[i];
        weights[i * INPSIZE + wsz + 1] = wfs[i];
        weights[i * INPSIZE + wsz + 2] = wfs[i + HIDSIZE];
    }
    let wl1 = Tensor::from_slice(&weights).view((HIDDENSIZE, INPUTSIZE));
    neuralnet::loadtensor(vs, "layer1.weight", &wl1);

    // layer1.bias
    let mut bias = [0.0f32 ; HIDDENSIZE as usize];
    bias.copy_from_slice(wdc);
    let wb1 = Tensor::from_slice(&bias).view(HIDDENSIZE);
    neuralnet::loadtensor(vs, "layer1.bias", &wb1);

    // layer2.weight
    let mut weights = [0.0f32 ; (HIDDENSIZE2 * HIDDENSIZE) as usize];
    weights.copy_from_slice(whdn);
    let wl2 = Tensor::from_slice(&weights).view((HIDDENSIZE2, HIDDENSIZE));
    neuralnet::loadtensor(vs, "layer2.weight", &wl2);

    // layer2.bias
    let mut bias = [0.0f32 ; HIDDENSIZE2 as usize];
    bias.copy_from_slice(wdc2);
    let wb1 = Tensor::from_slice(&bias).view(HIDDENSIZE2);
    neuralnet::loadtensor(vs, "layer2.bias", &wb1);

    // layer3.weight
    let mut weights = [0.0f32 ; HIDDENSIZE2 as usize];
    weights.copy_from_slice(whdn2);
    let wl2 = Tensor::from_slice(&weights).view((1, HIDDENSIZE2));
    neuralnet::loadtensor(vs, "layer3.weight", &wl2);

    // layer3.bias
    let bias = [wdc3 ; 1];
    let wb2 = Tensor::from_slice(&bias).view(1);
    neuralnet::loadtensor(vs, "layer3.bias", &wb2);

    Ok(())
}

pub fn storeweights(weights_dst : &mut weight::Weight, vs : VarStore, progress : usize) {
    println!("save to weights[{progress}]");

    // VarStore to weights
    let weights = vs.variables();
    let mut outp = [0.0f32 ; weight::N_WEIGHT];
    let mut tmp = [0.0f32 ; (INPUTSIZE * HIDDENSIZE) as usize];
    // let mut params = weight::EvalFile::V8.to_str().to_string() + "\n";

    let l1w = weights.get("layer1.weight").unwrap();
    // println!("layer1.weight:{:?}", l1w.size());
    let numel = l1w.numel();
    l1w.copy_data(tmp.as_mut_slice(), numel);
    for i in 0..HIDDENSIZE as usize {
        let wsz = bitboard::CELL_2D * 2;  // 先後の石の升分
        let offset_out = i * wsz;
        let offset = i * INPUTSIZE as usize;
        outp[offset_out..offset_out + wsz].copy_from_slice(
            &tmp[offset..offset + wsz]);
        outp[weight::N_WEIGHT_TEBAN + i] = tmp[wsz + offset];
        outp[weight::N_WEIGHT_FIXED_B + i] = tmp[wsz + 1 + offset];
        outp[weight::N_WEIGHT_FIXED_W + i] = tmp[wsz + 2 + offset];
    }

    let keys = [
        ("layer1.bias", weight::N_WEIGHT_INPUTBIAS),
        ("layer2.weight", weight::N_WEIGHT_LAYER1),
        ("layer2.bias", weight::N_WEIGHT_LAYER1BIAS),
        ("layer3.weight", weight::N_WEIGHT_LAYER2),
    ];
    for (key, offset) in keys {
        let l1w = weights.get(key).unwrap();
        // println!("{key}:{:?}", l1w.size());
        let numel = l1w.numel();
        l1w.copy_data(tmp.as_mut_slice(), numel);
        outp[offset..offset + numel].copy_from_slice(&tmp[0..numel]);
    }

    let l3b = weights.get("layer3.bias").unwrap();
    // println!("layer3.bias:{:?}", l3b.size());
    let numel = l3b.numel();
    l1w.copy_data(tmp.as_mut_slice(), numel);
    *outp.last_mut().unwrap() = tmp[0];

    println!("store to weight [{progress}]");
    weights_dst.copy_from_slice(&outp, progress);
}

pub fn writeweights(weights : &weight::Weight) {
    println!("save to weights.txt");
    if let Err(err) = weights.write_latest("weights.txt") {
        panic!("{err}");
    }
}
