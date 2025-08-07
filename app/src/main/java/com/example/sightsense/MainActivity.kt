package com.example.sightsense

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.WindowManager
import androidx.core.content.ContextCompat
import com.example.sightsense.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.*
import kotlin.math.pow
import kotlin.math.sqrt
import androidx.core.graphics.createBitmap

@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    lateinit var labels:List<String>
    var colors = listOf(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    val paint = Paint()
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap:Bitmap
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    private lateinit var handler: Handler
    private lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    lateinit var model:SsdMobilenetV11Metadata1
    private var tts: TextToSpeech? = null
    private var captureSession: CameraCaptureSession? = null
    private var captureRequestBuilder: CaptureRequest.Builder? = null
    
    // TTS queue management
    private val ttsQueue = mutableListOf<String>()
    private var isTtsSpeaking = false
    private var lastDirectionalWarning = ""
    private var lastDirectionalWarningTime = 0L
    private val directionalWarningCooldown = 1500L // 1.5 seconds between warnings
    
    // Directional warning thresholds
    private val closeDistanceThreshold = 0.25f
    private val centerZoneThreshold = 0.15f
    
    // Binary contrast toggle
    private var isBinaryMode = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tts = TextToSpeech(this, this)
        setContentView(R.layout.activity_main)
        
        // Keep screen on while using camera
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        
        getPermission()

        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = SsdMobilenetV11Metadata1.newInstance(this)
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        imageView = findViewById(R.id.imageView)

        textureView = findViewById(R.id.textureView)
        
        // Add tap gesture to toggle binary mode
        val mainLayout = findViewById<androidx.constraintlayout.widget.ConstraintLayout>(R.id.main)
        mainLayout.setOnClickListener {
            isBinaryMode = !isBinaryMode
            Log.d("BinaryMode", "Toggled to: $isBinaryMode")
        }
        
        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                openCamera()
            }
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
                updatePreview()
                configureTransform()
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                
                // Apply binary contrast if screen is tapped
                if (isBinaryMode) {
                    bitmap = convertToBinary(bitmap)
                }
                
                var image = TensorImage.fromBitmap(bitmap)
                image = imageProcessor.process(image)

                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray
                outputs.numberOfDetectionsAsTensorBuffer.floatArray

                val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width
                paint.textSize = h/15f
                paint.strokeWidth = h/85f
                
                var closestObjectIndex = -1
                var bestDistanceScore = 0f
                
                var x: Int
                scores.forEachIndexed { index, fl ->
                    x = index
                    x *= 4
                    if(fl > 0.5){
                        // Calculate bounding box dimensions
                        val boxWidth = (locations[x+3] - locations[x+1]) * w
                        val boxHeight = (locations[x+2] - locations[x]) * h
                        val centerX = (locations[x+1] + locations[x+3]) * w / 2
                        val centerY = (locations[x] + locations[x+2]) * h / 2
                        
                        // Calculate distance score
                        val distanceScore = calculateDistanceScore(boxWidth, boxHeight, centerX, centerY, w.toFloat(), h.toFloat())
                        if (distanceScore > bestDistanceScore) {
                            bestDistanceScore = distanceScore
                            closestObjectIndex = index
                        }
                        
                        paint.color = colors[index]
                        paint.style = Paint.Style.STROKE
                        canvas.drawRect(RectF(locations[x+1] *w, locations[x] *h, locations[x+3] *w, locations[x+2] *h), paint)
                        paint.style = Paint.Style.FILL
                        canvas.drawText(labels[classes[index].toInt()] +" "+fl.toString(), locations[x+1] *w, locations[x] *h, paint)
                    }
                }
                
                // Announce the closest object with cooldown
                if (closestObjectIndex != -1) {
                    val closestLabel = labels[classes[closestObjectIndex].toInt()]
                    val currentTime = System.currentTimeMillis()
                    
                    // Calculate position and distance for the closest object
                    val closestX = closestObjectIndex * 4
                    val boxWidth = (locations[closestX+3] - locations[closestX+1]) * w
                    val boxHeight = (locations[closestX+2] - locations[closestX]) * h
                    val centerX = (locations[closestX+1] + locations[closestX+3]) * w / 2
                    val centerY = (locations[closestX] + locations[closestX+2]) * h / 2
                    calculateDistanceScore(boxWidth, boxHeight, centerX, centerY, w.toFloat(), h.toFloat())
                    val areaRatio = (boxWidth * boxHeight) / (w * h)

                    if (areaRatio > closeDistanceThreshold) {
                        val directionalWarning = generateDirectionalWarning(centerX, centerY, w.toFloat(), h.toFloat(), areaRatio, closestLabel)

                        if (directionalWarning != lastDirectionalWarning && 
                            currentTime - lastDirectionalWarningTime > directionalWarningCooldown) {
                            obstacleWarning(directionalWarning)
                            lastDirectionalWarning = directionalWarning
                            lastDirectionalWarningTime = currentTime
                        }
                    }
                }

                imageView.setImageBitmap(mutable)
            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

    }

    override fun onDestroy() {
        super.onDestroy()
        if (tts != null) {
            tts!!.stop()
            tts!!.shutdown()
        }
        model.close()
        closeCamera()
    }

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        // Update camera preview when orientation changes
        updatePreview()
        configureTransform()
    }

    @SuppressLint("MissingPermission")
    fun openCamera(){
        val backCameraId = findBackCamera()
        if (backCameraId != null) {
            cameraManager.openCamera(backCameraId, object:CameraDevice.StateCallback(){
                override fun onOpened(p0: CameraDevice) {
                    cameraDevice = p0
                    createCameraPreview()
                }

                override fun onDisconnected(p0: CameraDevice) {
                    Log.d("Camera", "Camera disconnected")
                    closeCamera()
                }

                override fun onError(p0: CameraDevice, p1: Int) {
                    Log.e("Camera", "Camera error: $p1")
                    closeCamera()
                }
            }, handler)
        } else {
            Log.e("Camera", "No back camera found on this device")
        }
    }

    private fun createCameraPreview() {
        try {
            val surfaceTexture = textureView.surfaceTexture
            val surface = Surface(surfaceTexture)

            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder!!.addTarget(surface)

            cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                override fun onConfigured(p0: CameraCaptureSession) {
                    captureSession = p0
                    updatePreview()
                    configureTransform()
                }
                override fun onConfigureFailed(p0: CameraCaptureSession) {
                    Log.e("Camera", "Failed to configure camera session")
                }
            }, handler)
        } catch (e: Exception) {
            Log.e("Camera", "Error creating camera preview: ${e.message}")
        }
    }

    private fun updatePreview() {
        if (captureSession == null) return
        
        try {
            val rotation = windowManager.defaultDisplay.rotation
            captureRequestBuilder!!.set(CaptureRequest.JPEG_ORIENTATION, getOrientation(rotation))
            
            captureSession!!.setRepeatingRequest(captureRequestBuilder!!.build(), null, handler)
        } catch (e: Exception) {
            Log.e("Camera", "Error updating preview: ${e.message}")
        }
    }

    private fun getOrientation(rotation: Int): Int {
        val backCameraId = findBackCamera()
        if (backCameraId != null) {
            val characteristics = cameraManager.getCameraCharacteristics(backCameraId)
            val sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            
            return when (rotation) {
                Surface.ROTATION_0 -> sensorOrientation
                Surface.ROTATION_90 -> (sensorOrientation + 270) % 360
                Surface.ROTATION_180 -> (sensorOrientation + 180) % 360
                Surface.ROTATION_270 -> (sensorOrientation + 90) % 360
                else -> sensorOrientation
            }
        }
        return 90 // fallback
    }

    private fun closeCamera() {
        try {
            captureSession?.close()
            captureSession = null
            cameraDevice.close()
        } catch (e: Exception) {
            Log.e("Camera", "Error closing camera: ${e.message}")
        }
    }

    private fun configureTransform() {
        val backCameraId = findBackCamera() ?: return
        val characteristics = cameraManager.getCameraCharacteristics(backCameraId)
        val sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        val deviceRotation = windowManager.defaultDisplay.rotation
        
        val matrix = Matrix()
        val viewRect = RectF(0f, 0f, textureView.width.toFloat(), textureView.height.toFloat())
        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()
        
        // Calculate rotation angle based on device orientation
        val rotationAngle = when (deviceRotation) {
            Surface.ROTATION_0 -> sensorOrientation.toFloat()
            Surface.ROTATION_90 -> ((sensorOrientation + 270) % 360).toFloat()
            Surface.ROTATION_180 -> ((sensorOrientation + 180) % 360).toFloat()
            Surface.ROTATION_270 -> ((sensorOrientation + 90) % 360).toFloat()
            else -> sensorOrientation.toFloat()
        }
        
        // Rotate the matrix based on device orientation
        matrix.postRotate(rotationAngle, centerX, centerY)
        
        textureView.setTransform(matrix)
    }

    private fun findBackCamera(): String? {
        for (cameraId in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                return cameraId
            }
        }
        return null
    }

    fun obstacleWarning(prompt: String) {
        // Add to queue if TTS is currently speaking
        if (isTtsSpeaking) {
            if (!ttsQueue.contains(prompt)) {
                ttsQueue.add(prompt)
            }
            return
        }

        if (ttsQueue.isNotEmpty()) {
            if (!ttsQueue.contains(prompt)) {
                ttsQueue.add(prompt)
            }
            processTtsQueue()
            return
        }

        speakText(prompt)
    }
    
    private fun speakText(text: String) {
        isTtsSpeaking = true
        tts?.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                Log.d("TTS", "Started speaking: $text")
            }
            
            override fun onDone(utteranceId: String?) {
                Log.d("TTS", "Finished speaking: $text")
                isTtsSpeaking = false
                processTtsQueue()
            }
            
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                Log.e("TTS", "Error speaking: $text")
                isTtsSpeaking = false
                processTtsQueue()
            }
        })
        
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "utterance_${System.currentTimeMillis()}")
    }
    
    private fun processTtsQueue() {
        if (ttsQueue.isNotEmpty() && !isTtsSpeaking) {
            val nextText = ttsQueue.removeAt(0)
            speakText(nextText)
        }
    }
    
    private fun calculateDistanceScore(
        boxWidth: Float, 
        boxHeight: Float, 
        centerX: Float, 
        centerY: Float, 
        imageWidth: Float, 
        imageHeight: Float
    ): Float {
        val areaFactor = boxWidth * boxHeight / (imageWidth * imageHeight)

        val centerDistance = sqrt(
            (centerX - imageWidth / 2).toDouble().pow(2.0) +
                    (centerY - imageHeight / 2).toDouble().pow(2.0)
        ).toFloat()
        val maxDistance = sqrt(imageWidth.toDouble().pow(2.0) + imageHeight.toDouble().pow(2.0)).toFloat()
        val centerFactor = 1f - (centerDistance / maxDistance)

        return areaFactor * 0.7f + centerFactor * 0.3f
    }
    
    private fun generateDirectionalWarning(
        centerX: Float, 
        centerY: Float, 
        imageWidth: Float, 
        imageHeight: Float, 
        areaRatio: Float,
        objectLabel: String
    ): String {
        val centerXPercent = centerX / imageWidth
        centerY / imageHeight

        val horizontalDirection = when {
            centerXPercent < (0.5f - centerZoneThreshold) -> "left"
            centerXPercent > (0.5f + centerZoneThreshold) -> "right"
            else -> "center"
        }

        return when {
            areaRatio > 0.45f -> {
                when (horizontalDirection) {
                    "center" -> "STOP! Step back! $objectLabel ahead"
                    "left" -> "STOP! Move right! $objectLabel on left"
                    "right" -> "STOP! Move left! $objectLabel on right"
                    else -> "STOP! Step back! $objectLabel ahead"
                }
            }
            areaRatio > 0.35f -> {
                when (horizontalDirection) {
                    "center" -> "Step back! $objectLabel ahead"
                    "left" -> "Move right! $objectLabel on left"
                    "right" -> "Move left! $objectLabel on right"
                    else -> "Step back! $objectLabel ahead"
                }
            }
            areaRatio > 0.25f -> {
                when (horizontalDirection) {
                    "center" -> "Caution ahead! $objectLabel"
                    "left" -> "Move right! $objectLabel on left"
                    "right" -> "Move left! $objectLabel on right"
                    else -> "Caution ahead! $objectLabel"
                }
            }
            else -> "Caution! $objectLabel"
        }
    }

    private fun getPermission(){
        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
            getPermission()
        }
    }

    override fun onInit(p0: Int) {
        if (p0 == TextToSpeech.SUCCESS) {
            val result = tts!!.setLanguage(Locale.US)

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS","The Language not supported!")
            }
        }
    }

    private fun convertToBinary(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val binaryBitmap = createBitmap(width, height)

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        // Convert to binary (black and white)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val red = Color.red(pixel)
            val green = Color.green(pixel)
            val blue = Color.blue(pixel)
            
            // Calculate luminance (grayscale)
            val luminance = (0.299 * red + 0.587 * green + 0.114 * blue).toInt()
            
            // Apply threshold for binary conversion
            val binaryValue = if (luminance > 128) 255 else 0
            
            // Set pixel to black or white
            pixels[i] = Color.rgb(binaryValue, binaryValue, binaryValue)
        }
        
        binaryBitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return binaryBitmap
    }
}