package com.bloodtailor.myllmapp

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
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

    // Server settings
    private var serverBaseUrl by mutableStateOf("")
    private val client by lazy {
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)  // Increased timeout for streaming
            .build()
    }

    // Available models and model state
    private var availableModels = mutableStateListOf<String>()
    private var currentModelLoaded by mutableStateOf(false)
    private var currentModel by mutableStateOf<String?>(null)
    private var currentContextLength by mutableStateOf<Int?>(null)
    private var formattedPrompt by mutableStateOf<String?>(null)
    
    // Shared preferences name
    private val PREFS_NAME = "LLMAppPreferences"
    private val SERVER_URL_KEY = "server_url"
    private val DEFAULT_SERVER_URL = "http://192.168.50.46:5000"
    private val DEFAULT_CONTEXT_LENGTH = 2048

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load saved server URL from preferences
        val sharedPref = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        serverBaseUrl = sharedPref.getString(SERVER_URL_KEY, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL

        // Fetch available models when the app starts
        fetchAvailableModels()

        // Check if any model is already loaded
        checkModelStatus()

        setContent {
            LLMAppUI()
        }
    }

    private fun saveServerUrl(url: String) {
        val sharedPref = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        with(sharedPref.edit()) {
            putString(SERVER_URL_KEY, url)
            apply()
        }
        serverBaseUrl = url
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
                // Show error message on UI thread
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainActivity,
                        "Error loading models: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
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
                            currentContextLength = if (jsonResponse.isNull("context_length")) null
                            else jsonResponse.getInt("context_length")
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainActivity,
                        "Error checking model status: ${e.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    private fun loadModel(modelName: String, contextLength: Int? = null, callback: (Boolean, String) -> Unit) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val jsonObject = JSONObject()
                jsonObject.put("model", modelName)
                if (contextLength != null) {
                    jsonObject.put("context_length", contextLength)
                }
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
                            currentContextLength = if (jsonResponse.has("context_length")) 
                                jsonResponse.getInt("context_length") else contextLength
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
                            currentContextLength = null
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

    private fun formatPrompt(prompt: String, callback: (String) -> Unit) {
        if (currentModel == null) {
            callback("No model selected")
            return
        }
        
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val jsonObject = JSONObject()
                jsonObject.put("prompt", prompt)
                jsonObject.put("model", currentModel)
                
                val mediaType = "application/json; charset=utf-8".toMediaType()
                val requestBody = jsonObject.toString().toRequestBody(mediaType)
                
                val request = Request.Builder()
                    .url("$serverBaseUrl/format_prompt")
                    .post(requestBody)
                    .build()
                
                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful) {
                        val responseBody = response.body?.string() ?: "{}"
                        val jsonResponse = JSONObject(responseBody)
                        val formatted = jsonResponse.optString("formatted_prompt", "")
                        
                        withContext(Dispatchers.Main) {
                            callback(formatted)
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            callback("Error: Failed to format prompt (${response.code})")
                        }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    callback("Error: ${e.message}")
                }
            }
        }
    }

    private fun sendStreamingPromptToServer(prompt: String, systemPrompt: String, formattedPromptOverride: String? = null, callback: (String, String) -> Unit) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Create JSON request
                val jsonObject = JSONObject()
                jsonObject.put("prompt", prompt)
                jsonObject.put("system_prompt", systemPrompt)
                jsonObject.put("model", currentModel)
                jsonObject.put("stream", true)
                
                // Add formatted prompt if provided
                if (formattedPromptOverride != null) {
                    jsonObject.put("formatted_prompt", formattedPromptOverride)
                }
                
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

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun LLMAppUI() {
        var prompt by remember { mutableStateOf("") }
        var contextLengthInput by remember { mutableStateOf(currentContextLength?.toString() ?: "") }
        var response by remember { mutableStateOf("Response will appear here...") }
        var isLoading by remember { mutableStateOf(false) }
        var selectedModel by remember { mutableStateOf(availableModels.firstOrNull() ?: "") }
        var expandedModelMenu by remember { mutableStateOf(false) }
        var statusMessage by remember { mutableStateOf("") }
        var showFormattedPrompt by remember { mutableStateOf(false) }
        val localFormattedPrompt = remember { mutableStateOf(formattedPrompt ?: "") }

        // Settings dialog state
        var showSettingsDialog by remember { mutableStateOf(false) }
        var tempServerUrl by remember { mutableStateOf(serverBaseUrl) }

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
        
        // Update context length input when currentContextLength changes
        LaunchedEffect(currentContextLength) {
            if (currentContextLength != null) {
                contextLengthInput = currentContextLength.toString()
            }
        }
        
        // Get the context for copying to clipboard
        val context = LocalContext.current
        
        MaterialTheme {
            Scaffold(
                topBar = {
                    TopAppBar(
                        title = { Text("LLM App") },
                        actions = {
                            IconButton(onClick = { showSettingsDialog = true }) {
                                Icon(Icons.Default.Settings, contentDescription = "Settings")
                            }
                        }
                    )
                }
            ) { innerPadding ->
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Server Connection Info
                    Text(
                        "Server: $serverBaseUrl",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                    
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
                                expanded = expandedModelMenu,
                                onExpandedChange = { expandedModelMenu = !expandedModelMenu }
                            ) {
                                TextField(
                                    value = selectedModel,
                                    onValueChange = {},
                                    readOnly = true,
                                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedModelMenu) },
                                    modifier = Modifier
                                        .menuAnchor()
                                        .fillMaxWidth()
                                )

                                ExposedDropdownMenu(
                                    expanded = expandedModelMenu,
                                    onDismissRequest = { expandedModelMenu = false }
                                ) {
                                    availableModels.forEach { model ->
                                        DropdownMenuItem(
                                            text = { Text(model) },
                                            onClick = {
                                                selectedModel = model
                                                expandedModelMenu = false
                                            }
                                        )
                                    }
                                }
                            }
                        }
                    }
                    
                    // Context Length Row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Context:", modifier = Modifier.width(80.dp))
                        
                        OutlinedTextField(
                            value = contextLengthInput,
                            onValueChange = { 
                                // Only allow numeric input
                                if (it.isEmpty() || it.all { char -> char.isDigit() }) {
                                    contextLengthInput = it
                                }
                            },
                            label = { Text("Context Length") },
                            placeholder = { Text("Default: $DEFAULT_CONTEXT_LENGTH") },
                            singleLine = true,
                            modifier = Modifier.weight(1f)
                        )
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
                                // Parse context length if provided
                                val contextLength = if (contextLengthInput.isNotEmpty()) {
                                    contextLengthInput.toIntOrNull()
                                } else {
                                    null
                                }
                                
                                loadModel(selectedModel, contextLength) { success, message ->
                                    isLoading = false
                                    statusMessage = message
                                    
                                    // Get formatted prompt example if successful
                                    if (success && prompt.isNotEmpty()) {
                                        formatPrompt(prompt) { formatted ->
                                            formattedPrompt = formatted
                                            localFormattedPrompt.value = formatted
                                        }
                                    }
                                }
                            },
                            enabled = !isLoading && (!currentModelLoaded || (currentModel != selectedModel) 
                                || (contextLengthInput.toIntOrNull() != currentContextLength)),
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
                
                    // User prompt input
                    TextField(
                        value = prompt,
                        onValueChange = { 
                            prompt = it 
                            // Update formatted prompt preview if needed
                            if (currentModelLoaded && showFormattedPrompt) {
                                formatPrompt(it) { formatted ->
                                    formattedPrompt = formatted
                                    localFormattedPrompt.value = formatted
                                }
                            }
                        },
                        label = { Text("Enter your prompt") },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(120.dp)
                    )
                    
                    // Show formatted prompt toggle
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Start,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Checkbox(
                            checked = showFormattedPrompt,
                            onCheckedChange = { checked ->
                                showFormattedPrompt = checked
                                if (checked && prompt.isNotEmpty() && currentModelLoaded) {
                                    formatPrompt(prompt) { formatted ->
                                        formattedPrompt = formatted
                                        localFormattedPrompt.value = formatted
                                    }
                                }
                            }
                        )
                        Text("Show formatted prompt")
                    }
                    
                    // Formatted prompt preview
                    if (showFormattedPrompt) {
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp)
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Text("Formatted Prompt:", style = MaterialTheme.typography.labelMedium)
                                Spacer(modifier = Modifier.height(4.dp))
                                SelectionContainer {
                                    Text(
                                        text = localFormattedPrompt.value,
                                        fontFamily = FontFamily.Monospace,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .background(
                                                Color(0xFFF5F5F5),
                                                shape = RoundedCornerShape(4.dp)
                                            )
                                            .padding(8.dp)
                                    )
                                }
                            }
                        }
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Button(
                            onClick = {
                                if (prompt.isNotEmpty() && currentModelLoaded) {
                                    isLoading = true
                                    response = "Generating response..."
                                    sendStreamingPromptToServer(prompt, "", null) { status, result ->
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
                            enabled = !isLoading && prompt.isNotEmpty() && currentModelLoaded
                        ) {
                            Text("Send")
                        }
                    }

                    if (isLoading) {
                        LinearProgressIndicator(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp)
                        )
                    }

                    // Response area with scrolling and copy button
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f)
                            .padding(vertical = 8.dp)
                    ) {
                        Card(
                            modifier = Modifier.fillMaxSize()
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(2.dp)
                            ) {
                                // Response text with selection support
                                SelectionContainer(
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .padding(start = 12.dp, top = 12.dp, end = 12.dp, bottom = 48.dp)
                                ) {
                                    Text(
                                        text = response,
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .verticalScroll(rememberScrollState())
                                    )
                                }
                                
                                // Copy button at the bottom right
                                FloatingActionButton(
                                    onClick = {
                                        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                        val clip = ClipData.newPlainText("LLM Response", response)
                                        clipboard.setPrimaryClip(clip)
                                        
                                        // Show toast notification
                                        Toast.makeText(context, "Response copied to clipboard", Toast.LENGTH_SHORT).show()
                                    },
                                    modifier = Modifier
                                        .align(Alignment.BottomEnd)
                                        .padding(8.dp)
                                        .size(40.dp),
                                    containerColor = MaterialTheme.colorScheme.primaryContainer
                                ) {
                                    Icon(
                                        Icons.Default.ContentCopy,
                                        contentDescription = "Copy to clipboard",
                                        tint = MaterialTheme.colorScheme.onPrimaryContainer
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // Settings Dialog
            if (showSettingsDialog) {
                AlertDialog(
                    onDismissRequest = { showSettingsDialog = false },
                    title = { Text("Server Settings") },
                    text = {
                        Column {
                            Text("Enter the server URL including port (e.g., http://192.168.1.100:5000)")
                            Spacer(modifier = Modifier.height(8.dp))
                            OutlinedTextField(
                                value = tempServerUrl,
                                onValueChange = { tempServerUrl = it },
                                label = { Text("Server URL") },
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
                    },
                    confirmButton = {
                        Button(
                            onClick = {
                                saveServerUrl(tempServerUrl)
                                showSettingsDialog = false
                                fetchAvailableModels()  // Refresh models list
                                checkModelStatus()      // Check model status with new URL
                            }
                        ) {
                            Text("Save")
                        }
                    },
                    dismissButton = {
                        TextButton(
                            onClick = {
                                tempServerUrl = serverBaseUrl  // Reset to current value
                                showSettingsDialog = false
                            }
                        ) {
                            Text("Cancel")
                        }
                    }
                )
            }
        }
    }
}