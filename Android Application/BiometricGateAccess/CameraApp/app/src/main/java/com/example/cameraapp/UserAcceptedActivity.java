package com.example.cameraapp;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import okhttp3.*;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;

public class UserAcceptedActivity extends AppCompatActivity {

    private boolean isDoorLocked = false;

    private ImageView gateIcon;
    private ImageView doorIcon;
    private Button gateButton;
    private Button doorButton;

    private static final String BASE_URL = "http://192.168.227.2:5000";  // Base URL for Raspberry Pi server
    private OkHttpClient client = new OkHttpClient(); // HTTP client for making requests

    @SuppressLint("CutPasteId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_accepted);
        // Initialize the UI components
        gateIcon = findViewById(R.id.gateIcon);
        doorIcon = findViewById(R.id.doorIcon);
        gateButton = findViewById(R.id.gateButton);
        doorButton = findViewById(R.id.doorButton);

        fetchStatus();  // Fetch the initial status from the Raspberry Pi server

        // Set up listeners for the gate and door buttons
        gateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toggleGate(); // Toggle the gate state when the button is clicked
            }
        });

        doorButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toggleDoor(); // Toggle the door state when the button is clicked
            }
        });
    }

    private void fetchStatus() {
        // Create a request to fetch the status from the server
        Request request = new Request.Builder()
                .url(BASE_URL + "/get_status")
                .build();
        // Asynchronously execute the request
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                e.printStackTrace();
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    try {
                        // Parse the response to get the status
                        JSONObject jsonObject = new JSONObject(response.body().string());
                        isDoorLocked = jsonObject.getBoolean("isDoorLocked");

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                // Update the UI with the fetched status
                                updateGateUI();
                                updateDoorUI();
                            }
                        });
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    private void toggleGate() {
        // Toggle the gate status and update the UI
        isDoorLocked = !isDoorLocked;
        updateGateUI();
        // Create a request to toggle the gate state on the server
        RequestBody body = RequestBody.create(null, new byte[]{});
        Request request = new Request.Builder()
                .url(BASE_URL + "/toggle_gate")
                .post(body)
                .build();
        // Asynchronously execute the request
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                e.printStackTrace();
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                Log.d("ToggleGate", "Gate toggled");
            }
        });
    }

    private void toggleDoor() {
        // Toggle the door status and update the UI
        isDoorLocked = !isDoorLocked;
        updateDoorUI();

        // Create a JSON object to send the door status
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("isDoorLocked", isDoorLocked);
        } catch (JSONException e) {
            e.printStackTrace();
        }

        // Create a request to toggle the door state on the server
        RequestBody body = RequestBody.create(jsonObject.toString(), MediaType.get("application/json; charset=utf-8"));
        Request request = new Request.Builder()
                .url(BASE_URL + "/toggle_door")
                .post(body)
                .build();
        // Asynchronously execute the request
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                e.printStackTrace();
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                Log.d("ToggleDoor", "Door toggled");
            }
        });
    }

    // Update the UI based on the gate status
    private void updateGateUI() {
        if (isDoorLocked) {
            gateIcon.setImageResource(R.drawable.gate); // Set image resource for gate
            gateButton.setText("Open / Close Gate");
        } else {
            gateIcon.setImageResource(R.drawable.gate); // Update button text
            gateButton.setText("Open / Close Gate");
        }
    }

    private void updateDoorUI() {
        if (isDoorLocked) {
            doorIcon.setImageResource(R.drawable.door_locked); // Set image resource for gate
            doorButton.setText("Unlock Door");
        } else {
            doorIcon.setImageResource(R.drawable.door_unlocked); // Update button text
            doorButton.setText("Lock Door");
        }
    }

}
