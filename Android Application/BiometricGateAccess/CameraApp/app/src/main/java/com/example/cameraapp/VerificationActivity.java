package com.example.cameraapp;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.exifinterface.media.ExifInterface;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class VerificationActivity extends AppCompatActivity {
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private CameraSelector cameraSelector;
    private ProcessCameraProvider cameraProvider;
    private PreviewView previewView;
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;
    private int imagesUploadedCount = 0;
    private TextView messageTextView;
    private TextView timerTextView;
    private RelativeLayout timerLayout;
    private MediaPlayer mediaPlayer;
    private View flashOverlay;
    private int number_of_verification_calls = 0;
    private ProgressDialog progressDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_verification);
        // Initialize UI elements
        previewView = findViewById(R.id.previewView);
        Button switchCameraButton = findViewById(R.id.switchCameraButton);
        Button takePicturesButton = findViewById(R.id.takePicturesButton);
        messageTextView = findViewById(R.id.messageTextView);
        timerTextView = findViewById(R.id.timerTextView);
        timerLayout = findViewById(R.id.timerLayout);
        flashOverlay = findViewById(R.id.flashOverlay); // View for flash effect

        switchCameraButton.setOnClickListener(v -> switchCamera()); // Switch camera on button click

        takePicturesButton.setOnClickListener(v -> startPhotoCaptureSequence()); // Start photo capture sequence on button click

        // Initially hide the timer layout
        timerLayout.setVisibility(View.GONE);

        // Initialize MediaPlayer for camera click sound
        mediaPlayer = MediaPlayer.create(this, R.raw.camera_click);

        // Check and request camera permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        } else {
            startCamera(); // Start the camera if permissions are already granted
        }

        cameraExecutor = Executors.newSingleThreadExecutor(); // Initialize camera executor
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera(); // Start the camera if permission is granted
            } else {
                // do nothing
            }
        }
    }

    private void startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this); // Get the camera provider instance
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get(); // Get the camera provider
                cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT) // Set the camera to the front-facing camera
                        .build();
                bindPreview(cameraProvider);  // Bind the camera preview
            } catch (ExecutionException | InterruptedException e) {
                // Exception
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build(); // Create a Preview object
        imageCapture = new ImageCapture.Builder().build(); // Create an ImageCapture object
        preview.setSurfaceProvider(previewView.getSurfaceProvider()); // Set the surface provider for the preview
        cameraProvider.unbindAll(); // Unbind any previously bound use cases
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture); // Bind the preview and image capture use cases
    }

    private void switchCamera() {
        if (cameraSelector == null) {
            return;
        }
        @SuppressLint("RestrictedApi") int newLensFacing = (cameraSelector.getLensFacing() == CameraSelector.LENS_FACING_BACK)
                ? CameraSelector.LENS_FACING_FRONT
                : CameraSelector.LENS_FACING_BACK; // Toggle between front and back cameras

        cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(newLensFacing)
                .build();
        bindPreview(cameraProvider); // Re-bind the camera preview with the new camera selector
    }

    private void startPhotoCaptureSequence() {
        // Show the timer layout
        timerLayout.setVisibility(View.VISIBLE);

        // Start a countdown of 5 seconds before taking photos
        new CountDownTimer(6000, 1000) {
            public void onTick(long millisUntilFinished) {
                // Update the timer TextView each second
                timerTextView.setText(String.valueOf(millisUntilFinished / 1000));
            }

            public void onFinish() {
                // Hide the timer layout and capture the photos
                timerLayout.setVisibility(View.GONE);
                takePictures();
            }
        }.start();
    }

    private void takePictures() {
        if (imageCapture == null) {
            return;
        }

        imagesUploadedCount = 0;

        Handler handler = new Handler(Looper.getMainLooper());

        for (int i = 0; i < 3; i++) {
            int delay = i * 1500; // 1.5 seconds delay between each picture

            handler.postDelayed(() -> {
                // Show flash effect and play camera sound before taking the picture
                showFlashEffect();
                playCameraSound();

                File photoFile = new File(getExternalFilesDir(null), "photo" + System.currentTimeMillis() + ".jpeg");
                ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(photoFile).build();
                // Capture the image
                imageCapture.takePicture(outputOptions, ContextCompat.getMainExecutor(this), new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        Log.d("CameraXApp", "Photo capture succeeded: " + photoFile.getAbsolutePath());
                        resizeAndUploadImage(photoFile); // Upload the captured image
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e("CameraXApp", "Photo capture failed: " + exception.getMessage(), exception);
                    }
                });
            }, delay);
        }
    }



    private void showFlashEffect() {
        if (flashOverlay != null) {
            flashOverlay.setVisibility(View.VISIBLE);
            flashOverlay.animate()
                    .alpha(0f)
                    .setDuration(300)
                    .withEndAction(() -> {
                        flashOverlay.setVisibility(View.GONE);
                        flashOverlay.setAlpha(1f);
                    })
                    .start();
        }
    }

    private void playCameraSound() {
        if (mediaPlayer != null) {
            mediaPlayer.start(); // Play the camera shutter sound
        }
    }

    private void resizeAndUploadImage(File file) {
        try {
            // Read the image as Bitmap
            Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath());

            // Read EXIF data to determine image orientation
            ExifInterface exif = new ExifInterface(file.getAbsolutePath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            int rotationInDegrees = exifToDegrees(orientation);

            // Rotate the image if necessary
            Bitmap rotatedBitmap = rotateBitmap(bitmap, rotationInDegrees);
            Bitmap resizedBitmap = rotatedBitmap;

            // Upload the image
            File resizedFile = new File(getExternalFilesDir(null), file.getName());
            try (FileOutputStream out = new FileOutputStream(resizedFile)) {
                resizedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            }

            uploadImage(resizedFile);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private int exifToDegrees(int exifOrientation) {
        // Convert EXIF orientation to degrees for rotation
        if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_90) {
            return 90;
        } else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_180) {
            return 180;
        } else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_270) {
            return 270;
        }
        return 0;
    }

    private Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    private void uploadImage(File file) {
        // Initialize and configure the progress dialog
        progressDialog = new ProgressDialog(this);
        progressDialog.setMessage("Verification in progress...");
        progressDialog.setCancelable(false); // Prevents dismissing the dialog until response is received
        progressDialog.show(); // Show the loading spinner

        number_of_verification_calls = number_of_verification_calls + 1; // Increment the verification call counter
        System.out.println(number_of_verification_calls);
        String androidID = Settings.Secure.getString(getContentResolver(), Settings.Secure.ANDROID_ID);


        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.MINUTES) // Set connection timeout
                .writeTimeout(5, TimeUnit.MINUTES)   // Set write timeout
                .readTimeout(5, TimeUnit.MINUTES)    // Set read timeout
                .build();


        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", file.getName(),
                        RequestBody.create(MediaType.parse("image/jpeg"), file))
                .addFormDataPart("android_id", androidID)
                .addFormDataPart("number_of_verification_calls", String.valueOf(number_of_verification_calls))
                .build();

        Request request = new Request.Builder()
                .url("http://192.168.227.1:5000/verification")
                .post(requestBody)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                Log.e("Upload", "Image upload failed: " + e.getMessage(), e);
                if (progressDialog.isShowing()) {
                    progressDialog.dismiss();
                }
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        progressDialog.dismiss();
                        Toast.makeText(VerificationActivity.this, "Image upload failed: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                });
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (progressDialog.isShowing()) {
                    progressDialog.dismiss();
                }
                if (!response.isSuccessful()) {
                    Log.e("Upload", "Image upload failed: " + response.message());
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            progressDialog.dismiss();
                            Toast.makeText(VerificationActivity.this, "REJECTED", Toast.LENGTH_SHORT).show();
                            try {
                                Thread.sleep(2500);
                                finish();
                            } catch (InterruptedException e) {
                                throw new RuntimeException(e);
                            }
                        }
                    });
                } else {
                    Log.d("Upload", "Image upload succeeded: " + response.message());
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            imagesUploadedCount++;
                            if (imagesUploadedCount == 3) {
                                Intent intent = new Intent(VerificationActivity.this, UserAcceptedActivity.class);
                                startActivity(intent); // Start UserAcceptedActivity if all 3 images pass all 3 modules and are accepted: the modules are Face Detection, Anti-Spoofing, and Face Recognition.
                                finish();
                            }
                        }
                    });
                }
            }
        });
        if (number_of_verification_calls == 3) { // Reset the counter after 3 verification calls
            number_of_verification_calls = 0;
        }

    }

    @Override
    protected void onPause() {
        super.onPause();
        if (progressDialog != null && progressDialog.isShowing()) {
            progressDialog.dismiss();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
        // Dismiss the ProgressDialog if the activity is destroyed (e.g., when navigating away)
        if (progressDialog != null && progressDialog.isShowing()) {
            progressDialog.dismiss();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Dismiss the ProgressDialog if the activity is resumed (e.g., when navigating away)
        if (progressDialog != null && progressDialog.isShowing()) {
            progressDialog.dismiss();
        }
    }


}
