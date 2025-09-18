/*
 * ESP32 PD Neural Prediction - Store Results & Send on Website Request
 * Stores Jetson evaluation results until website requests them
 * Displays on LCD and sends to speaker 10 seconds after website transmission
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <SPI.h>
#include "pd_tbr_data.h"

#ifndef LED_BUILTIN
#define LED_BUILTIN 2
#endif

#define TFT_CS     5
#define TFT_RST    22
#define TFT_DC     21

#define GRAY_COLOR    0x7BEF
#define DARK_GRAY     0x4208

// WiFi credentials - UPDATE THESE
const char* ssid = "OnePlus Nord CE 3 Lite 5G";
const char* password = "murugangeetha27";

// Jetson Nano target for auto-send
const char* jetson_ip = "10.240.134.227";
const int jetson_port = 8888;

WebServer server(80);
Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

const int SCREEN_WIDTH = 128;
const int SCREEN_HEIGHT = 160;

int requestCounter = 0;
int websiteRequestCounter = 0;
unsigned long serverStartTime = 0;
bool jetsonConnected = false;
bool websiteConnected = false;
String transmissionLog[10];
int logIndex = 0;
bool logFull = false;
const int LED_PIN = LED_BUILTIN;

// Timing for LCD display after website transmission
unsigned long websiteTransmissionTime = 0;
bool waitingForLCDDisplay = false;
const unsigned long LCD_DISPLAY_DELAY = 5000; // 10 seconds

// HTTP client for sending to Jetson
WiFiClient jetsonClient;

struct StoredEvaluationResults {
  String classification;
  float confidence;
  int request_id;
  float theta;
  float low_beta;
  float high_beta;
  float tbr;
  String timestamp;
  bool has_results;
  String recommendations;
};

StoredEvaluationResults storedResults;
bool resultsAvailable = false;
bool resultsSentToWebsite = false;

void setup() {
  Serial.begin(115200);
  delay(1000);
 
  Serial.println(F("ESP32 PD Neural Prediction Server"));
 
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
 
  digitalWrite(LED_PIN, HIGH);
  delay(100);
  digitalWrite(LED_PIN, LOW);
 
  // Initialize stored results
  storedResults.has_results = false;
  resultsAvailable = false;
  resultsSentToWebsite = false;
  waitingForLCDDisplay = false;
 
  initializeLCD();
  randomSeed(analogRead(0));
  connectToWiFi();
  setupWebServer();
  serverStartTime = millis();
  displayStartupInfo();
  updateLCDStatus("System Ready", "Waiting for data...");
 
  Serial.println(F("ESP32 PD Neural Prediction Server ready!"));
  Serial.println(F("Website endpoint: /random-eeg (sends to Jetson & Website)"));
  Serial.println(F("Results endpoint: /get_results (for website)"));
  Serial.println(F("Prediction storage: /prediction_result"));
  Serial.printf("Jetson target: %s:%d\n", jetson_ip, jetson_port);
}

void loop() {
  server.handleClient();
  
  // Check if it's time to display results on LCD after website transmission
  if (waitingForLCDDisplay && (millis() - websiteTransmissionTime >= LCD_DISPLAY_DELAY)) {
    displayStoredResults();
    sendToSpeaker(); // Trigger speaker at same time as LCD display
    waitingForLCDDisplay = false;
  }
 
  static unsigned long lastWiFiCheck = 0;
  if (millis() - lastWiFiCheck > 30000) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println(F("WiFi disconnected. Reconnecting..."));
      updateLCDStatus("WiFi Lost", "Reconnecting...");
      connectToWiFi();
    }
    lastWiFiCheck = millis();
  }
 
  static unsigned long lastLCDUpdate = 0;
  if (millis() - lastLCDUpdate > 5000) {
    if (!resultsAvailable && !waitingForLCDDisplay) {
      updateLCDTransmissionLog();
    }
    lastLCDUpdate = millis();
  }
 
  delay(10);
}

void initializeLCD() {
  Serial.println(F("Initializing LCD..."));
 
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(0);
  tft.fillScreen(ST77XX_BLACK);
 
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(10, 20);
  tft.println("PD Neural");
  tft.setCursor(5, 40);
  tft.println("Predict");
 
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_CYAN);
  tft.setCursor(10, 70);
  tft.println("Starting...");
 
  Serial.println(F("LCD initialized"));
  delay(2000);
}

void updateLCDStatus(String status, String details) {
  tft.fillScreen(ST77XX_BLACK);
 
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_CYAN);
  tft.setCursor(5, 5);
  tft.println("PD Neural Prediction");
  tft.drawLine(5, 15, SCREEN_WIDTH - 5, 15, ST77XX_CYAN);
 
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 25);
  tft.print("Status: ");
  tft.setTextColor(ST77XX_GREEN);
  tft.println(status);
 
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 40);
  tft.println(details);
 
  // Request counters
  tft.setTextColor(ST77XX_YELLOW);
  tft.setCursor(5, 60);
  tft.print("Website: ");
  tft.println(websiteRequestCounter);
 
  tft.setCursor(5, 75);
  tft.print("Stored: ");
  tft.println(resultsAvailable ? "YES" : "NO");
 
  tft.setCursor(5, 90);
  tft.print("Uptime: ");
  tft.print((millis() - serverStartTime) / 1000);
  tft.println("s");
 
  if (WiFi.status() == WL_CONNECTED) {
    tft.setTextColor(ST77XX_GREEN);
    tft.setCursor(5, 105);
    tft.print("WiFi: Connected");
   
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(5, 120);
    tft.print("IP: ");
    tft.println(WiFi.localIP().toString());
  } else {
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(5, 105);
    tft.println("WiFi: Disconnected");
  }
}

void displayDataSentWaiting(int requestNum) {
  tft.fillScreen(ST77XX_BLACK);
  
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_CYAN);
  tft.setCursor(5, 5);
  tft.println("PD Neural Prediction");
  tft.drawLine(5, 15, SCREEN_WIDTH - 5, 15, ST77XX_CYAN);
  
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_GREEN);
  
  String dataText = "DATA " + String(requestNum);
  int dataWidth = dataText.length() * 12;
  int dataStartX = (SCREEN_WIDTH - dataWidth) / 2;
  if (dataStartX < 5) dataStartX = 5;
  
  tft.setCursor(dataStartX, 50);
  tft.println(dataText);
  
  tft.setTextSize(2);
  tft.setCursor(25, 80);
  tft.print("sent");
  
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_YELLOW);
  tft.setCursor(20, 120);
  tft.println("Waiting...");
}

void displayStoredResults() {
  tft.fillScreen(ST77XX_BLACK);
 
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_CYAN);
  tft.setCursor(5, 5);
  tft.println("PD Neural Prediction");
  tft.drawLine(5, 15, SCREEN_WIDTH - 5, 15, ST77XX_CYAN);
 
  // Data number - large and centered
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_WHITE);
  
  String dataText = "DATA " + String(storedResults.request_id);
  int dataWidth = dataText.length() * 12;
  int dataStartX = (SCREEN_WIDTH - dataWidth) / 2;
  if (dataStartX < 5) dataStartX = 5;
  
  tft.setCursor(dataStartX, 35);
  tft.println(dataText);
  
  // Results label
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_GREEN);
  tft.setCursor(5, 70);
  tft.println("Results:");
  
  // Classification - full text on one line
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 85);
  tft.println(storedResults.classification);
  
  // Confidence
  tft.setTextColor(ST77XX_YELLOW);
  tft.setCursor(5, 105);
  tft.print("Confidence: ");
  tft.setTextColor(ST77XX_WHITE);
  tft.print(storedResults.confidence, 1);
  tft.println("%");
  
  // Small status info at bottom
  tft.setTextSize(1);
  tft.setTextColor(DARK_GRAY);
  tft.setCursor(5, SCREEN_HEIGHT - 12);
  tft.print("Total: ");
  tft.print(requestCounter);
}

void sendToSpeaker() {
  // Placeholder for speaker/voice output
  // Implement with your audio module (e.g., DFPlayer Mini, or TTS module)
  Serial.println("🔊 VOICE OUTPUT:");
  Serial.print("   Severity: ");
  Serial.println(storedResults.classification);
  Serial.print("   Confidence: ");
  Serial.print(storedResults.confidence, 1);
  Serial.println(" percent");
 
  // You can add actual speaker control here
  // Example: dfPlayer.play(getSeverityAudioFile(storedResults.classification));
}

void addTransmissionToLog(String logEntry) {
  transmissionLog[logIndex] = logEntry;
  logIndex++;
  if (logIndex >= 10) {
    logIndex = 0;
    logFull = true;
  }
  Serial.println("LCD: " + logEntry);
}

void updateLCDTransmissionLog() {
  tft.fillScreen(ST77XX_BLACK);
 
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_CYAN);
  tft.setCursor(5, 5);
  tft.println("PD Neural Prediction");
  tft.drawLine(5, 15, SCREEN_WIDTH - 5, 15, ST77XX_CYAN);
 
  tft.setTextColor(ST77XX_WHITE);
  int yPos = 25;
  int startIdx, endIdx;
 
  if (logFull) {
    startIdx = logIndex;
    endIdx = logIndex + 10;
  } else {
    startIdx = 0;
    endIdx = logIndex;
  }
 
  int displayCount = 0;
  for (int i = startIdx; i < endIdx && displayCount < 6; i++) {
    int idx = i % 10;
    if (transmissionLog[idx].length() > 0) {
      tft.setCursor(5, yPos);
      if (displayCount == 0) {
        tft.setTextColor(ST77XX_GREEN);
        tft.print("> ");
      } else {
        tft.setTextColor(ST77XX_WHITE);
        tft.print("  ");
      }
      tft.println(transmissionLog[idx]);
      yPos += 12;
      displayCount++;
    }
  }
 
  // Status indicators
  tft.setTextColor(ST77XX_YELLOW);
  tft.setCursor(5, 110);
  tft.print("W:");
  tft.print(websiteRequestCounter);
  tft.print(" S:");
  tft.println(resultsAvailable ? "Y" : "N");
 
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 125);
  if (resultsAvailable) {
    tft.print("Results stored");
  } else {
    tft.print("Waiting...");
  }
}

// Send data to Jetson Nano
bool sendDataToJetson(PDRecord randomData, int requestId) {
  Serial.printf("Sending data to Jetson Nano at %s:%d\n", jetson_ip, jetson_port);
 
  try {
    if (jetsonClient.connect(jetson_ip, jetson_port)) {
      DynamicJsonDocument doc(400);
      doc["source"] = "esp32_auto_send";
      doc["request_id"] = requestId;
      doc["timestamp"] = millis();
      doc["record_id"] = randomData.record_id;
      doc["theta"] = randomData.theta;
      doc["high_beta"] = randomData.high_beta;
      doc["low_beta"] = randomData.low_beta;
      doc["tbr"] = randomData.tbr;
      doc["data_type"] = "pd_tbr_classification";
     
      String jsonData;
      serializeJson(doc, jsonData);
     
      jetsonClient.print("POST /receive_data HTTP/1.1\r\n");
      jetsonClient.print("Host: ");
      jetsonClient.print(jetson_ip);
      jetsonClient.print("\r\n");
      jetsonClient.print("Content-Type: application/json\r\n");
      jetsonClient.print("Content-Length: ");
      jetsonClient.print(jsonData.length());
      jetsonClient.print("\r\n\r\n");
      jetsonClient.print(jsonData);
     
      unsigned long timeout = millis() + 5000;
      while (jetsonClient.available() == 0 && millis() < timeout) {
        delay(1);
      }
     
      String response = "";
      while (jetsonClient.available()) {
        response += jetsonClient.readString();
      }
     
      jetsonClient.stop();
     
      Serial.println("Successfully sent data to Jetson Nano");
     
      if (!jetsonConnected) {
        jetsonConnected = true;
        blinkConnectionEstablished();
      }
     
      return true;
     
    } else {
      Serial.println("Failed to connect to Jetson Nano");
      jetsonConnected = false;
      return false;
    }
  } catch (...) {
    Serial.println("Exception occurred while sending to Jetson");
    jetsonConnected = false;
    jetsonClient.stop();
    return false;
  }
}

// Website endpoint for EEG data (sends to Jetson, sends to website)
void handleRandomEEG() {
  server.sendHeader(F("Access-Control-Allow-Origin"), "*");
  server.sendHeader(F("Access-Control-Allow-Methods"), F("GET, POST, OPTIONS"));
  server.sendHeader(F("Access-Control-Allow-Headers"), F("Content-Type"));
 
  if (!websiteConnected) {
    websiteConnected = true;
    updateLCDStatus("Website Connected!", "Sending data...");
    delay(1000);
  }
 
  PDRecord randomData = getRandomPDRecord();
 
  if (randomData.record_id == -1) {
    server.send(500, F("application/json"), F("{\"error\":\"Failed to retrieve PD data\"}"));
    Serial.println(F("Website request failed"));
    return;
  }
 
  websiteRequestCounter++;
 
  // Reset stored results for new evaluation
  resultsAvailable = false;
  resultsSentToWebsite = false;
  storedResults.has_results = false;
  waitingForLCDDisplay = false;
 
  // Display "DATA X sent" on LCD immediately when website requests
  displayDataSentWaiting(websiteRequestCounter);
 
  // Send data to Jetson Nano
  bool jetsonSent = sendDataToJetson(randomData, websiteRequestCounter);
 
  // Format data for website (without results yet)
  DynamicJsonDocument doc(400);
  doc["success"] = true;
  doc["dataIndex"] = randomData.record_id;
  doc["data"]["theta"] = randomData.theta;
  doc["data"]["lowBeta"] = randomData.low_beta;
  doc["data"]["highBeta"] = randomData.high_beta;
  doc["data"]["tbr"] = randomData.tbr;
  doc["data"]["severity"] = "Processing...";
  doc["timestamp"] = millis();
  doc["source"] = "esp32_embedded";
  doc["jetson_sent"] = jetsonSent;
  doc["status"] = "data_sent_to_jetson_evaluating";
 
  String jsonResponse;
  serializeJson(doc, jsonResponse);
 
  server.send(200, F("application/json"), jsonResponse);
 
  blinkDataTransmitted();
 
  String logEntry = "WEB " + String(websiteRequestCounter) + " → Jetson";
  addTransmissionToLog(logEntry);
 
  Serial.print(F("Website Request #"));
  Serial.print(websiteRequestCounter);
  Serial.println(F(" - Data sent to Jetson, waiting for evaluation..."));
}

// Endpoint for website to get stored evaluation results
void handleGetResults() {
  server.sendHeader(F("Access-Control-Allow-Origin"), "*");
  server.sendHeader(F("Access-Control-Allow-Methods"), F("GET, POST, OPTIONS"));
  server.sendHeader(F("Access-Control-Allow-Headers"), F("Content-Type"));
 
  DynamicJsonDocument doc(800);
 
  if (resultsAvailable && storedResults.has_results && !resultsSentToWebsite) {
    // Send results to website automatically (no LCD display animation)
    doc["success"] = true;
    doc["results_available"] = true;
    doc["classification"] = storedResults.classification;
    doc["confidence"] = storedResults.confidence;
    doc["risk_score"] = storedResults.confidence; // Risk score = confidence
    doc["severity"] = storedResults.classification;
    doc["data"]["theta"] = storedResults.theta;
    doc["data"]["lowBeta"] = storedResults.low_beta;
    doc["data"]["highBeta"] = storedResults.high_beta;
    doc["data"]["tbr"] = storedResults.tbr;
    doc["timestamp"] = storedResults.timestamp;
    doc["request_id"] = storedResults.request_id;
    doc["source"] = "esp32_stored_jetson_evaluation";
   
    // Generate treatment recommendations based on severity
    String recommendations = generateTreatmentRecommendations(storedResults.classification, storedResults.confidence);
    doc["recommendations"] = recommendations;
   
    // Mark as sent to website and start 10-second timer for LCD display
    resultsSentToWebsite = true;
    websiteTransmissionTime = millis();
    waitingForLCDDisplay = true;
   
    Serial.println("🌐 RESULTS SENT TO WEBSITE:");
    Serial.println("  Classification: " + storedResults.classification);
    Serial.println("  Confidence: " + String(storedResults.confidence) + "%");
    Serial.println("  ⏰ LCD display and speaker will activate in 10 seconds");
   
    addTransmissionToLog("SENT " + String(storedResults.request_id) + " to web");
   
  } else if (resultsSentToWebsite) {
    // Results already sent, return the same data
    doc["success"] = true;
    doc["results_available"] = true;
    doc["classification"] = storedResults.classification;
    doc["confidence"] = storedResults.confidence;
    doc["risk_score"] = storedResults.confidence;
    doc["severity"] = storedResults.classification;
    doc["data"]["theta"] = storedResults.theta;
    doc["data"]["lowBeta"] = storedResults.low_beta;
    doc["data"]["highBeta"] = storedResults.high_beta;
    doc["data"]["tbr"] = storedResults.tbr;
    doc["timestamp"] = storedResults.timestamp;
    doc["request_id"] = storedResults.request_id;
    doc["source"] = "esp32_stored_jetson_evaluation";
    
    String recommendations = generateTreatmentRecommendations(storedResults.classification, storedResults.confidence);
    doc["recommendations"] = recommendations;
    
  } else {
    doc["success"] = false;
    doc["results_available"] = false;
    doc["status"] = "no_results_stored";
    doc["message"] = "No evaluation results available yet. Please fill data first.";
   
    Serial.println("❌ Website requested results but none are stored");
  }
 
  String jsonResponse;
  serializeJson(doc, jsonResponse);
 
  server.send(200, F("application/json"), jsonResponse);
}

String generateTreatmentRecommendations(String severity, float confidence) {
  String recommendations = "";
 
  if (severity == "Normal") {
    recommendations = "• Continue regular health monitoring\n";
    recommendations += "• Maintain active lifestyle with regular exercise\n";
    recommendations += "• Follow balanced diet rich in antioxidants\n";
    recommendations += "• Schedule routine neurological check-ups annually";
  } else if (severity.indexOf("Mild") >= 0) {
    recommendations = "• Start with Levodopa/Carbidopa (25/100mg) twice daily\n";
    recommendations += "• Begin physical therapy focusing on mobility\n";
    recommendations += "• Implement speech therapy if needed\n";
    recommendations += "• Regular exercise: walking, swimming, tai chi\n";
    recommendations += "• Schedule neurologist visits every 3-6 months";
  } else if (severity.indexOf("Moderate") >= 0) {
    recommendations = "• Adjust Levodopa dosage (may increase to 3-4 times daily)\n";
    recommendations += "• Consider adding Dopamine agonists\n";
    recommendations += "• Intensive physical therapy and occupational therapy\n";
    recommendations += "• Address sleep disorders and depression if present\n";
    recommendations += "• Regular medication timing is crucial";
  } else if (severity.indexOf("Severe") >= 0) {
    recommendations = "• Optimize complex medication regimen with neurologist\n";
    recommendations += "• Consider advanced therapies: DBS evaluation\n";
    recommendations += "• Comprehensive care team needed\n";
    recommendations += "• Address non-motor symptoms: cognitive, psychiatric\n";
    recommendations += "• Implement fall prevention strategies";
  } else {
    recommendations = "• Specialized palliative care evaluation\n";
    recommendations += "• Advanced directive and end-of-life planning\n";
    recommendations += "• Complex medication management\n";
    recommendations += "• 24/7 nursing care may be required\n";
    recommendations += "• Focus on comfort and dignity of care";
  }
 
  if (confidence < 70) {
    recommendations += "\n• Low confidence - require comprehensive clinical evaluation";
  } else if (confidence >= 90) {
    recommendations += "\n• High confidence - proceed with recommended treatment plan";
  }
 
  return recommendations;
}

// Store evaluation results from Jetson Nano (modified to NOT display on LCD immediately)
void handlePredictionResult() {
  server.sendHeader(F("Access-Control-Allow-Origin"), "*");
  server.sendHeader(F("Access-Control-Allow-Methods"), F("GET, POST, OPTIONS"));
  server.sendHeader(F("Access-Control-Allow-Headers"), F("Content-Type"));
 
  if (server.method() == HTTP_POST) {
    String requestBody = server.arg("plain");
   
    DynamicJsonDocument doc(800);
    DeserializationError error = deserializeJson(doc, requestBody);
   
    if (error) {
      server.send(400, F("application/json"), F("{\"error\":\"Invalid JSON\"}"));
      Serial.println(F("Failed to parse prediction JSON"));
      return;
    }
   
    // Store all prediction results (don't display on LCD yet)
    storedResults.classification = doc["classification"].as<String>();
    storedResults.confidence = doc["confidence"].as<float>();
    storedResults.request_id = doc["request_id"].as<int>();
    storedResults.theta = doc["theta"].as<float>();
    storedResults.low_beta = doc["low_beta"].as<float>();
    storedResults.high_beta = doc["high_beta"].as<float>();
    storedResults.tbr = doc["tbr"].as<float>();
    storedResults.timestamp = doc["timestamp"].as<String>();
    storedResults.has_results = true;
   
    resultsAvailable = true;
    resultsSentToWebsite = false; // Not sent to website yet
   
    Serial.println(F("💾 EVALUATION RESULTS STORED FROM JETSON:"));
    Serial.print(F("  Classification: "));
    Serial.println(storedResults.classification);
    Serial.print(F("  Confidence: "));
    Serial.print(storedResults.confidence);
    Serial.println(F("%"));
    Serial.print(F("  Request ID: "));
    Serial.println(storedResults.request_id);
    Serial.println(F("  📦 Results stored, waiting for website request"));
    Serial.println(F("  📺 NOT displaying on LCD yet"));
   
    server.send(200, F("application/json"), F("{\"status\":\"results_stored\"}"));
   
    addTransmissionToLog("STORED " + String(storedResults.request_id) + " from AI");
   
    // DO NOT display on LCD - keep current display unchanged
   
  } else {
    server.send(405, F("text/plain"), F("Method not allowed"));
  }
}

void blinkLED(int times, int duration = 200) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(duration);
    digitalWrite(LED_PIN, LOW);
    delay(duration);
  }
}

void blinkConnectionEstablished() {
  Serial.println(F("LED: Connection established - 3 blinks"));
  blinkLED(3, 300);
}

void blinkDataTransmitted() {
  Serial.println(F("LED: Data transmitted - single blink"));
  blinkLED(1, 150);
}

void connectToWiFi() {
  updateLCDStatus("Connecting", "WiFi...");
 
  WiFi.begin(ssid, password);
  Serial.print(F("Connecting to WiFi"));
 
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(1000);
    Serial.print(".");
    attempts++;
   
    if (attempts % 5 == 0) {
      updateLCDStatus("Connecting", "WiFi..." + String(attempts));
    }
  }
 
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println(F("WiFi Connected!"));
    Serial.print(F("IP: "));
    Serial.println(WiFi.localIP());
   
    updateLCDStatus("Connected", WiFi.localIP().toString());
    delay(2000);
  } else {
    Serial.println();
    Serial.println(F("WiFi Failed"));
    updateLCDStatus("WiFi Failed", "Check credentials");
  }
}

void setupWebServer() {
  server.onNotFound([]() {
    if (server.method() == HTTP_OPTIONS) {
      server.sendHeader(F("Access-Control-Allow-Origin"), "*");
      server.sendHeader(F("Access-Control-Allow-Methods"), F("GET, POST, OPTIONS"));
      server.sendHeader(F("Access-Control-Allow-Headers"), F("Content-Type"));
      server.send(200, F("text/plain"), "");
    } else {
      server.sendHeader(F("Access-Control-Allow-Origin"), "*");
      server.send(404, F("text/plain"), F("Not found"));
    }
  });
 
  // Website integration endpoints
  server.on(F("/random-eeg"), HTTP_GET, handleRandomEEG);
  server.on(F("/random-eeg"), HTTP_OPTIONS, handleCORS);
 
  // Results endpoint for website
  server.on(F("/get_results"), HTTP_GET, handleGetResults);
  server.on(F("/get_results"), HTTP_OPTIONS, handleCORS);
 
  // Jetson Nano prediction results storage
  server.on(F("/prediction_result"), HTTP_POST, handlePredictionResult);
  server.on(F("/prediction_result"), HTTP_OPTIONS, handleCORS);
 
  // Status and info endpoints
  server.on(F("/status"), HTTP_GET, handleStatusRequest);
  server.on(F("/status"), HTTP_OPTIONS, handleCORS);
  server.on("/", HTTP_GET, handleRootRequest);
  server.on(F("/health"), HTTP_GET, []() {
    server.sendHeader(F("Access-Control-Allow-Origin"), "*");
    server.send(200, F("text/plain"), F("OK"));
  });
 
  server.begin();
  Serial.print(F("Web server started at: http://"));
  Serial.println(WiFi.localIP());
  Serial.println(F("Website endpoint: /random-eeg"));
  Serial.println(F("Results endpoint: /get_results"));
  Serial.println(F("Storage endpoint: /prediction_result"));
}

void handleCORS() {
  server.sendHeader(F("Access-Control-Allow-Origin"), "*");
  server.sendHeader(F("Access-Control-Allow-Methods"), F("GET, POST, OPTIONS"));
  server.sendHeader(F("Access-Control-Allow-Headers"), F("Content-Type"));
  server.send(200, F("text/plain"), "");
}

void handleStatusRequest() {
  server.sendHeader(F("Access-Control-Allow-Origin"), "*");
 
  unsigned long uptime = millis() - serverStartTime;
 
  DynamicJsonDocument doc(600);
  doc["status"] = "online";
  doc["data_source"] = "embedded_progmem";
  doc["data_type"] = "pd_tbr_classification";
  doc["total_pd_records"] = TOTAL_PD_RECORDS;
  doc["website_requests"] = websiteRequestCounter;
  doc["uptime_ms"] = uptime;
  doc["uptime_seconds"] = uptime / 1000;
  doc["free_heap"] = ESP.getFreeHeap();
  doc["ip_address"] = WiFi.localIP().toString();
  doc["jetson_connected"] = jetsonConnected;
  doc["website_connected"] = websiteConnected;
  doc["jetson_target"] = String(jetson_ip) + ":" + String(jetson_port);
  doc["results_stored"] = resultsAvailable;
  doc["results_ready"] = resultsAvailable;
  doc["results_sent_to_website"] = resultsSentToWebsite;
  doc["waiting_for_lcd_display"] = waitingForLCDDisplay;
 
  if (resultsAvailable) {
    doc["stored_classification"] = storedResults.classification;
    doc["stored_confidence"] = storedResults.confidence;
  }
 
  String response;
  serializeJson(doc, response);
 
  server.send(200, F("application/json"), response);
 
  Serial.println(F("Status request served"));
}

void handleRootRequest() {
  String html = F("<!DOCTYPE html><html><head><title>ESP32 PD Neural Prediction Server</title>");
  html += F("<style>body{font-family:Arial;margin:40px;background:#f5f5f5;}");
  html += F(".container{background:white;padding:30px;border-radius:10px;}");
  html += F(".status{background:#e8f5e8;padding:15px;margin:10px 0;border-radius:5px;}");
  html += F(".stored{background:#fff3cd;padding:10px;margin:10px 0;border-radius:5px;}");
  html += F("</style></head>");
  html += F("<body><div class='container'><h1>ESP32 PD Neural Prediction Server</h1>");
  html += F("<div class='status'><h3>Server Status: Online ✅</h3></div>");
 
  if (resultsAvailable) {
    html += F("<div class='stored'><h4>💾 Results Stored from Jetson AI</h4>");
    html += F("<p><strong>Classification:</strong> ");
    html += storedResults.classification;
    html += F("</p><p><strong>Confidence:</strong> ");
    html += String(storedResults.confidence, 1);
    html += F("%</p>");
    
    if (resultsSentToWebsite) {
      html += F("<p><strong>Status:</strong> Sent to website");
      if (waitingForLCDDisplay) {
        unsigned long remaining = (LCD_DISPLAY_DELAY - (millis() - websiteTransmissionTime)) / 1000;
        html += F(" - LCD display in ");
        html += String(remaining);
        html += F(" seconds");
      } else {
        html += F(" - Displayed on LCD");
      }
      html += F("</p>");
    } else {
      html += F("<p><strong>Status:</strong> Ready for website request</p>");
    }
    html += F("</div>");
  } else {
    html += F("<div class='stored'><h4>⏳ No Results Stored</h4>");
    html += F("<p>Waiting for Jetson AI evaluation</p></div>");
  }
 
  html += F("<p><strong>Website Requests:</strong> ");
  html += String(websiteRequestCounter);
  html += F("</p>");
 
  html += F("<h3>Endpoints:</h3>");
  html += F("<ul>");
  html += F("<li><a href='/random-eeg'>/random-eeg</a> - Send data to Jetson for evaluation</li>");
  html += F("<li><a href='/get_results'>/get_results</a> - Get stored evaluation results</li>");
  html += F("<li><a href='/status'>/status</a> - Server status JSON</li>");
  html += F("</ul>");
 
  html += F("</div></body></html>");
 
  server.send(200, F("text/html"), html);
  Serial.println(F("Root page served"));
}

void displayStartupInfo() {
  Serial.println(F("PD Neural Prediction Server Information:"));
  Serial.print(F("   Total Records: "));
  Serial.println(TOTAL_PD_RECORDS);
 
  Serial.println(F("Features:"));
  Serial.println(F("   - Website integration (/random-eeg)"));
  Serial.println(F("   - Results storage from Jetson AI"));
  Serial.println(F("   - Website results delivery (/get_results)"));
  Serial.println(F("   - LCD display with 10-second delay after website transmission"));
  Serial.println(F("   - Voice/speaker output synchronized with LCD"));
  Serial.println(F("   - LED feedback"));
  Serial.println(F("   - Automatic result transmission to website"));
}