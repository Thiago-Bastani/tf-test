const tf = require("@tensorflow/tfjs");

const modelo = tf.sequential();
const inputLayer = tf.layers.dense({ units: 1, inputShape: [1] });
modelo.add(inputLayer);
modelo.compile({ loss: "meanSquaredError", optimizer: "sgd" });

const x = tf.tensor([[1], [2], [3], [4]]);
const y = tf.tensor([[102], [203], [304], [405]]);

modelo.fit(x, y, { epochs: 100000 }).then(() => {
  console.log(modelo.predict(tf.tensor([5])).dataSync());
});
