package com.simpledataflow.nothotdogapptutorial

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.*
import com.simpledataflow.nothotdogapptutorial.ml.Model
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    // Constants for permissions
    companion object {
        const val REQUEST_CODE_PERMISSIONS = 5
        const val REQUIRED_PERMISSION = Manifest.permission.CAMERA
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (hasPermission()) {
            // .post ensures that camera is started
            // only if view was initialized (displayed on the UI)
            view_finder.post { startCamera() }
        } else {
            requestPermission()
        }

        view_finder.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateTransform() // handling rotations
        }
    }

    /**
     * Camera
     */
    // For background ANALYSIS
    private val executor = Executors.newSingleThreadExecutor()

    private fun startCamera() {
        // 1. PREVIEW USE CASE (camera stream). Display video stream
        // Create configuration object for PREVIEW use case
        val previewConfig = PreviewConfig.Builder().apply {
            setTargetResolution(Size(640, 480))
        }.build()


        // Build preview use case
        val preview = Preview(previewConfig)

        // Every time view_finder is updated, recompute layout
        preview.setOnPreviewOutputUpdateListener {

            // To update the SurfaceTexture, we have to remove it and re-add it
            val parent = view_finder.parent as ViewGroup
            parent.removeView(view_finder)
            parent.addView(view_finder, 0)

            view_finder.surfaceTexture = it.surfaceTexture
            updateTransform()
        }

        // 2. ANALYSIS USE CASE. Continuously analyze the images from video that's coming into the camera
        // Create configuration object for ANALYSIS use case
        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            setImageReaderMode(
                ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE) // analyze latest images (not every image)
        }.build()

        // run classifier
        val analyzer = ImageAnalysis.Analyzer { image: ImageProxy, _: Int ->
            val bitmap = image.toBitmap()
            processImage(bitmap)

        }

        // Build ANALYSIS use case
        val analyzerUseCase = ImageAnalysis(analyzerConfig).apply {
            setAnalyzer(executor, analyzer)
        }

        // Bind use cases to lifecycle (if activity finishes camera also finishes)
        CameraX.bindToLifecycle(this, preview, analyzerUseCase)
    }

    private fun updateTransform() {
        val matrix = Matrix()

        // Compute the center of the view finder
        val centerX = view_finder.width / 2f
        val centerY = view_finder.height / 2f

        // Correct preview output to account for display rotation
        val rotationDegrees = when(view_finder.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }
        matrix.postRotate(-rotationDegrees.toFloat(), centerX, centerY)

        // Finally, apply transformations
        view_finder.setTransform(matrix)
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer // Y
        val uBuffer = planes[1].buffer // U
        val vBuffer = planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }


    /**
     * Classifying an image
     * @param bitmap    Android representation of an image
     */
    private fun processImage(bitmap: Bitmap) {
        try {
            // Tensorflow representation of the image
            var tfImage = TensorImage(DataType.FLOAT32)
            // Loading the original android image to the tensorflow image
            tfImage.load(bitmap)

            // Processing the image
            // The model we build uses 224x224 images as an input so we need so resize the image
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build()
            tfImage = imageProcessor.process(tfImage)

            // Load "model.tflite" "Model" refers to the name before ".tflite". So if model was named "mymodel.tflite" we would write MyModel.newInstance(...)
            val model = Model.newInstance(this@MainActivity)

            // Apply normalization operator for image classification (a necessary step)
            val probabilityProcessor =
                TensorProcessor.Builder().add(NormalizeOp(0f, 255f)).build()

            // running classification
            val outputs =
                model.process(probabilityProcessor.process(tfImage.tensorBuffer))
            // getting the output
            val outputBuffer = outputs.outputFeature0AsTensorBuffer

            // adding labels to the output
            val tensorLabel =
                TensorLabel(arrayListOf("hot_dog", "not_hot_dog"), outputBuffer)

            // getting the first label (hot dog) probability
            // if 80 (you can change that) then we are pretty sure it is a hotdog -> update UI
            val probability = tensorLabel.mapWithFloatValue["hot_dog"]
            probability?.let {
                if (it > 0.80) {
                    tv_result.text = "Hotdog!"
                } else {
                    tv_result.text = "Not Hotdog!"
                }
            }
            // Logs for debugging
            Log.d("sdf", "HOT DOG : " + probability)
        } catch (e: Exception) {
            Log.d("sdf", "Exception is " + e.localizedMessage)
        }
    }

    /**
     * Permissions
     */
    // checking if we have permission already
    private fun hasPermission(): Boolean =
        checkSelfPermission(REQUIRED_PERMISSION) == PackageManager.PERMISSION_GRANTED

    // requesting permission
    private fun requestPermission() =
        requestPermissions(arrayOf(REQUIRED_PERMISSION), REQUEST_CODE_PERMISSIONS)

    // after user granted (or not) checking if permission was granted
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (hasPermission()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT)
                    .show()
                finish()
            }
        }
    }
}


