package com.bloodtailor.myllmapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException

class MainActivity : ComponentActivity() {

    // Change this to your PC's IP address and port
    private val serverUrl = "http://192.168.50.46:5000/query"
    private val client = OkHttpClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            LLMAppUI()
        }
    }

    @Composable
    fun LLMAppUI() {
        var prompt by remember { mutableStateOf("") }
        var response by remember { mutableStateOf("Response will appear here...") }
        var isLoading by remember { mutableStateOf(false) }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            TextField(
                value = prompt,
                onValueChange = { prompt = it },
                label = { Text("Enter your prompt") },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(120.dp)
            )

            Button(
                onClick = {
                    if (prompt.isNotEmpty()) {
                        isLoading = true
                        sendPromptToServer(prompt) { result ->
                            response = result
                            isLoading = false
                        }
                    }
                },
                modifier = Modifier.align(Alignment.CenterHorizontally),
                enabled = !isLoading && prompt.isNotEmpty()
            ) {
                Text("Send")
            }

            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.align(Alignment.CenterHorizontally)
                )
            }

            Text(
                text = response,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(vertical = 8.dp)
            )
        }
    }

    private fun sendPromptToServer(prompt: String, callback: (String) -> Unit) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Create JSON request
                val jsonObject = JSONObject()
                jsonObject.put("prompt", prompt)
                val jsonRequest = jsonObject.toString()

                // Prepare the request
                val mediaType = "application/json; charset=utf-8".toMediaType()
                val requestBody = jsonRequest.toRequestBody(mediaType)

                val request = Request.Builder()
                    .url(serverUrl)
                    .post(requestBody)
                    .build()

                // Execute the request
                client.newCall(request).enqueue(object : Callback {
                    override fun onFailure(call: Call, e: IOException) {
                        // Handle failure
                        lifecycleScope.launch(Dispatchers.Main) {
                            callback("Error: ${e.message}")
                        }
                    }

                    override fun onResponse(call: Call, response: Response) {
                        // Handle response
                        val responseBody = response.body?.string() ?: "No response"

                        lifecycleScope.launch(Dispatchers.Main) {
                            if (response.isSuccessful) {
                                try {
                                    val jsonResponse = JSONObject(responseBody)
                                    val llmResponse = jsonResponse.optString("response", "No data found")
                                    callback(llmResponse)
                                } catch (e: Exception) {
                                    callback("Error parsing response: ${e.message}")
                                }
                            } else {
                                callback("Server error: ${response.code}")
                            }
                        }
                    }
                })

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    callback("Error: ${e.message}")
                }
            }
        }
    }
}
