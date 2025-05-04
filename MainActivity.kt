package com.bloodtailor.myllmapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import java.util.concurrent.TimeUnit

class MainActivity : ComponentActivity() {

    // Change this to your PC's IP address and port
    private val serverBaseUrl = "http://192.168.50.46:5000"
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)  // Increased timeout for streaming
        .build()

    // Available models and model state
    private var availableModels = mutableStateListOf<String>()
    private var currentModelLoaded by mutableStateOf(false)
    private var currentModel by mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Fetch available models when the app starts
        fetchAvailableModels()

        // Check if any model is already loaded
        checkModelStatus()

        setContent {
            LLMAppUI()
        }
    }

    private fun fetchAvailableModels() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$serverBaseUrl/models")
                    .get()
                    .build()

                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        val responseBody = response.body?.string() ?: "{}"
                        val jsonResponse = JSONObject(responseBody)
                        val models = jsonResponse.getJSONArray("models")

                        val modelsList = mutableListOf<String>()
                        for (i in 0 until models.length()) {
                            modelsList.add(models.getString(i))
                        }

                        withContext(Dispatchers.Main) {
                            availableModels.clear()
                            availableModels.addAll(modelsList)
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun checkModelStatus() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$serverBaseUrl/model/status")
                    .get()
                    .build()

                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        val responseBody = response.body?.string() ?: "{}"
                        val jsonResponse = JSONObject(responseBody)

                        withContext(Dispatchers.Main) {
                            currentModelLoaded = jsonResponse.getBoolean("loaded")
                            currentModel = if (jsonResponse.isNull("current_model")) null
                            else jsonResponse.getString("current_model")
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun loadModel(modelName: String, callback: (Boolean, String) -> Unit) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val jsonObject = JSONObject()
                jsonObject.put("model", modelName)
                val jsonRequest = jsonObject.toString()

                val mediaType = "application/json; charset=utf-8".toMediaType()
                val requestBody = jsonRequest.toRequestBody(mediaType)

                val request = Request.Builder()
                    .url("$serverBaseUrl/model/load")
                    .post(requestBody)
                    .build()

                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        val responseBody = response.body?.string() ?: "{}"
                        val jsonResponse = JSONObject(responseBody)

                        withContext(Dispatchers.Main) {
                            currentModelLoaded = true
                            currentModel = modelName
                            callback(true, jsonResponse.optString("message", "Model loaded successfully"))
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            callback(false, "Failed to load model: ${response.code}")
                        }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    callback(false, "Error: ${e.message}")
                }
            }
        }
    }

    private fun unloadModel(callback: (Boolean, String) -> Unit) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$serverBaseUrl/model/unload")
                    .post(RequestBody.create(null, byteArrayOf()))
                    .build()

                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        val responseBody = response.body?.string() ?: "{}"
                        val jsonResponse = JSONObject(responseBody)

                        withContext(Dispatchers.Main) {
                            currentModelLoaded = false
                            currentModel = null
                            callback(true, jsonResponse.optString("message", "Model unloaded successfully"))
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            callback(false, "Failed to unload model: ${response.code}")
                        }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    callback(false, "Error: ${e.message}")
                }
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun LLMAppUI() {
        var prompt by remember { mutableStateOf("") }
        var systemPrompt by remember { mutableStateOf("") }
        var response by remember { mutableStateOf("Response will appear here...") }
        var isLoading by remember { mutableStateOf(false) }
        var selectedModel by remember { mutableStateOf(availableModels.firstOrNull() ?: "") }
        var expanded by remember { mutableStateOf(false) }
        var statusMessage by remember { mutableStateOf("") }

        // Update selectedModel when availableModels changes
        LaunchedEffect(availableModels) {
            if (availableModels.isNotEmpty() && selectedModel.isEmpty()) {
                selectedModel = availableModels.first()
            }
        }

        // Update selectedModel when currentModel changes
        LaunchedEffect(currentModel) {
            if (currentModel != null && availableModels.contains(currentModel)) {
                selectedModel = currentModel!!
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Model Selection Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Model:", modifier = Modifier.width(80.dp))

                // Dropdown for model selection
                Box(
                    modifier = Modifier
                        .weight(1f)
                ) {
                    ExposedDropdownMenuBox(
                        expanded = expanded,
                        onExpandedChange = { expanded = !expanded }
                    ) {
                        TextField(
                            value = selectedModel,
                            onValueChange = {},
                            readOnly = true,
                            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                            modifier = Modifier
                                .menuAnchor()
                                .fillMaxWidth()
                        )

                        ExposedDropdownMenu(
                            expanded = expanded,
                            onDismissRequest = { expanded = false }
                        ) {
                            availableModels.forEach { model ->
                                DropdownMenuItem(
                                    text = { Text(model) },
                                    onClick = {
                                        selectedModel = model
                                        expanded = false
                                    }
                                )
                            }
                        }
                    }
                }
            }

            // Model control buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = {
                        isLoading = true
                        statusMessage = "Loading model..."
                        loadModel(selectedModel) { success, message ->
                            isLoading = false
                            statusMessage = message
                        }
                    },
                    enabled = !isLoading && !currentModelLoaded || (currentModel != selectedModel),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Load Model")
                }

                Button(
                    onClick = {
                        isLoading = true
                        statusMessage = "Unloading model..."
                        unloadModel { success, message ->
                            isLoading = false
                            statusMessage = message
                        }
                    },
                    enabled = !isLoading && currentModelLoaded,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Unload Model")
                }
            }

            // Status message
            if (statusMessage.isNotEmpty()) {
                Text(statusMessage, color = MaterialTheme.colorScheme.primary)
            }

            Divider(modifier = Modifier.padding(vertical = 8.dp))

            // System prompt input
            TextField(
                value = systemPrompt,
                onValueChange = { systemPrompt = it },
                label = { Text("System Prompt (optional)") },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(80.dp)
            )

            // User prompt input
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
                    if (prompt.isNotEmpty() && currentModelLoaded) {
                        isLoading = true
                        response = "Generating response..."
                        sendStreamingPromptToServer(prompt, systemPrompt) { status, result ->
                            if (status == "generating" || status == "complete") {
                                response = result
                            } else if (status == "error") {
                                response = "Error: $result"
                            }

                            if (status == "complete" || status == "error") {
                                isLoading = false
                            }
                        }
                    } else if (!currentModelLoaded) {
                        statusMessage = "Please load a model first"
                    }
                },
                modifier = Modifier.align(Alignment.CenterHorizontally),
                enabled = !isLoading && prompt.isNotEmpty() && currentModelLoaded
            ) {
                Text("Send")
            }

            if (isLoading) {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp)
                )
            }

            // Response area with scrolling
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(vertical = 8.dp)
            ) {
                Text(
                    text = response,
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
                )
            }
        }
    }

    private fun sendStreamingPromptToServer(prompt: String, systemPrompt: String, callback: (String, String) -> Unit) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Create JSON request
                val jsonObject = JSONObject()
                jsonObject.put("prompt", prompt)
                jsonObject.put("system_prompt", systemPrompt)
                jsonObject.put("model", currentModel)
                jsonObject.put("stream", true)
                val jsonRequest = jsonObject.toString()

                // Prepare the request
                val mediaType = "application/json; charset=utf-8".toMediaType()
                val requestBody = jsonRequest.toRequestBody(mediaType)

                val request = Request.Builder()
                    .url("$serverBaseUrl/query")
                    .post(requestBody)
                    .build()

                // Execute the request
                client.newCall(request).enqueue(object : Callback {
                    override fun onFailure(call: Call, e: IOException) {
                        // Handle failure
                        lifecycleScope.launch(Dispatchers.Main) {
                            callback("error", e.message ?: "Network error")
                        }
                    }

                    override fun onResponse(call: Call, response: Response) {
                        try {
                            if (!response.isSuccessful) {
                                lifecycleScope.launch(Dispatchers.Main) {
                                    callback("error", "Server error: ${response.code}")
                                }
                                return
                            }

                            // Read streaming responses
                            val responseBody = response.body
                            if (responseBody == null) {
                                lifecycleScope.launch(Dispatchers.Main) {
                                    callback("error", "Empty response")
                                }
                                return
                            }

                            responseBody.source().use { source ->
                                val buffer = okio.Buffer()
                                while (!source.exhausted()) {
                                    // Read a line
                                    source.readUtf8Line()?.let { line ->
                                        if (line.isNotEmpty()) {
                                            try {
                                                val jsonResponse = JSONObject(line)
                                                val status = jsonResponse.optString("status", "")

                                                when (status) {
                                                    "processing" -> {
                                                        // Initial response, nothing to do
                                                    }
                                                    "generating" -> {
                                                        val partial = jsonResponse.optString("partial", "")
                                                        lifecycleScope.launch(Dispatchers.Main) {
                                                            callback("generating", partial)
                                                        }
                                                    }
                                                    "complete" -> {
                                                        val fullResponse = jsonResponse.optString("response", "")
                                                        lifecycleScope.launch(Dispatchers.Main) {
                                                            callback("complete", fullResponse)
                                                        }
                                                    }
                                                    "error" -> {
                                                        val error = jsonResponse.optString("error", "Unknown error")
                                                        lifecycleScope.launch(Dispatchers.Main) {
                                                            callback("error", error)
                                                        }
                                                    }
                                                }
                                            } catch (e: Exception) {
                                                // Skip invalid JSON lines
                                            }
                                        }
                                    }
                                }
                            }

                        } catch (e: Exception) {
                            lifecycleScope.launch(Dispatchers.Main) {
                                callback("error", "Error processing response: ${e.message}")
                            }
                        } finally {
                            response.close()
                        }
                    }
                })

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    callback("error", "Error: ${e.message}")
                }
            }
        }
    }
}
