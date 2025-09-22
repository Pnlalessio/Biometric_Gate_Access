package com.example.cameraapp;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;

public class HomeActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        Button verificationButton = findViewById(R.id.btnVerification);
        Button registrationButton = findViewById(R.id.btnRegistration);

        // Set an OnClickListener for the verificationButton
        verificationButton.setOnClickListener(v -> {
            // Create an Intent to start VerificationActivity
            Intent intent = new Intent(HomeActivity.this, VerificationActivity.class);
            // Start the VerificationActivity
            startActivity(intent);
        });

        // Set an OnClickListener for the registrationButton
        registrationButton.setOnClickListener(v -> {
            // Create an Intent to start RegistrationActivity
            Intent intent = new Intent(HomeActivity.this, RegistrationActivity.class);
            // Start the RegistrationActivity
            startActivity(intent);
        });
    }
}



