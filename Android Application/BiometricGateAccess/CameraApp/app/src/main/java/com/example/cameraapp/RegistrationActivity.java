package com.example.cameraapp;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.provider.Settings;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class RegistrationActivity extends AppCompatActivity {
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    // Lists to store the URIs of the captured photos and ImageView references
    private List<Uri> photoUris;
    private List<ImageView> imageViews;
    // Index to keep track of the current photo being captured
    private int currentPhotoIndex = 0;
    // Button to trigger taking pictures
    private Button takePicturesButton;
    // Counter for failed registrations
    private int bad_registration_count = 0;
    private ProgressDialog progressDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_registration);

        // Check and request camera permission if not already granted
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        } else {
            // Permission already granted
        }

        photoUris = new ArrayList<>();
        imageViews = new ArrayList<>();

        imageViews.add(findViewById(R.id.photo1));
        imageViews.add(findViewById(R.id.photo2));
        imageViews.add(findViewById(R.id.photo3));

        // Set up the button to either take pictures or register based on the current photo index
        takePicturesButton = findViewById(R.id.takePicturesButton);
        takePicturesButton.setOnClickListener(v -> {
            if (currentPhotoIndex < 3) {
                dispatchTakePictureIntent();
            } else {
                takePicturesButton.setText("Register");
                takePicturesButton.setOnClickListener(v2 -> {
                    cropAndSendImages();
                });
            }
        });
    }

    // Starts the intent to capture an image using the device's camera
    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
            if (photoFile != null) {
                // Get a URI for the photo file and start the camera intent
                Uri photoUri = FileProvider.getUriForFile(this,
                        "com.example.myapp.fileprovider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        }
    }

    // Creates a temporary file to store the captured image
    private File createImageFile() throws IOException {
        String imageFileName = "photo" + currentPhotoIndex;
        File storageDir = getExternalFilesDir(null);
        File image = File.createTempFile(
                imageFileName,
                ".jpeg",
                storageDir
        );
        photoUris.add(Uri.fromFile(image));
        return image;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // Get the URI of the captured photo and update the corresponding ImageView
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Uri photoUri = photoUris.get(currentPhotoIndex);
            imageViews.get(currentPhotoIndex).setImageURI(photoUri);
            currentPhotoIndex++;

            // If all 3 photos have been taken (if they are a face), change the button text and set up the click listener for registration
            if (currentPhotoIndex == 3) {
                takePicturesButton.setText("Register");
                takePicturesButton.setOnClickListener(v -> {
                    cropAndSendImages();
                });
            }
        }
    }

    private void cropAndSendImages() {
        List<File> imageFiles = new ArrayList<>();

        for (int i = 0; i < photoUris.size(); i++) {
            Uri uri = photoUris.get(i);
            Bitmap bitmap = null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (bitmap != null) {
                // Processes each captured image: rotate based on EXIF data and store the image in the list
                try {
                    ExifInterface exif = new ExifInterface(uri.getPath());
                    int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
                    int degrees = exifToDegrees(orientation);
                    bitmap = rotateBitmap(bitmap, degrees);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                // Create a temporary file for each image
                File file = new File(getCacheDir(), "photo" + System.currentTimeMillis() + ".jpeg");
                try (FileOutputStream out = new FileOutputStream(file)) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                    imageFiles.add(file); // Add the file to the list to be sent
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        // Send all the images to the server in a single request
        sendImagesToServer(imageFiles);
    }


    private void sendImagesToServer(List<File> imageFiles) {
        // Initialize and configure the progress dialog
        progressDialog = new ProgressDialog(this);
        progressDialog.setMessage("Registration in progress...");
        progressDialog.setCancelable(false); // Prevents dismissing the dialog until response is received
        progressDialog.show(); // Show the loading spinner
        String androidID = Settings.Secure.getString(getContentResolver(), Settings.Secure.ANDROID_ID);

        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.MINUTES) // Maximum time to establish a connection
                .writeTimeout(5, TimeUnit.MINUTES)   // Maximum time to send data
                .readTimeout(5, TimeUnit.MINUTES)    // Maximum time to read the response
                .build();

        MultipartBody.Builder requestBodyBuilder = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("android_id", androidID);

        // Add all the images to the request 
        for (int i = 0; i < imageFiles.size(); i++) {
            File file = imageFiles.get(i);
            requestBodyBuilder.addFormDataPart("image" + i, file.getName(),
                    RequestBody.create(MediaType.parse("image/jpeg"), file));
        }

        RequestBody requestBody = requestBodyBuilder.build();

        Request request = new Request.Builder()
                .url("http://192.168.227.1:5000/registration")
                .post(requestBody)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                e.printStackTrace();
                if (progressDialog.isShowing()) {
                    progressDialog.dismiss();
                }
                runOnUiThread(() -> {
                    Toast.makeText(RegistrationActivity.this, "Failed to register images", Toast.LENGTH_SHORT).show();
                    resetRegistration();
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (progressDialog.isShowing()) {
                    progressDialog.dismiss();
                }
                if (response.isSuccessful()) {
                    runOnUiThread(() -> {
                        if (bad_registration_count == 0) {
                            Toast.makeText(RegistrationActivity.this, "You registered successfully!", Toast.LENGTH_SHORT).show();
                            try {
                                Thread.sleep(2500);
                                finish();
                            } catch (InterruptedException e) {
                                throw new RuntimeException(e);
                            }
                        }
                    });
                } else {
                    bad_registration_count += 1;
                    runOnUiThread(() -> {
                        Toast.makeText(RegistrationActivity.this, "Face not found. Repeat the registration.", Toast.LENGTH_SHORT).show();
                        resetRegistration();
                    });
                }
            }
        });
    }


    // Resets the registration process
    private void resetRegistration() {
        photoUris.clear();

        // Reset the ImageViews to their initial gray placeholder
        for (ImageView imageView : imageViews) {
            imageView.setImageDrawable(null);  // Remove the current image
            imageView.setBackgroundResource(R.drawable.photo_placeholder);  // Set the initial background
        }

        // Reset the photo index
        currentPhotoIndex = 0;

        // Reset the button text and click listener
        takePicturesButton.setText("Take Pictures");
        takePicturesButton.setOnClickListener(v -> {
            if (currentPhotoIndex < 3) {
                dispatchTakePictureIntent();
            }
        });

        // Reset the bad registration count
        bad_registration_count = 0;
    }

    // Converts EXIF orientation to rotation degrees
    private int exifToDegrees(int exifOrientation) {
        if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_90) {
            return 90;
        } else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_180) {
            return 180;
        } else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_270) {
            return 270;
        }
        return 0;
    }

    // Rotates the bitmap image by the specified degrees
    private Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }
}


