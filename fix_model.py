import tensorflow as tf

print("Loading old model...")
model = tf.keras.models.load_model("plant_model.h5", compile=False)

print("Re-saving in new format...")
model.save("plant_model_fixed.keras")

print("DONE ✅")