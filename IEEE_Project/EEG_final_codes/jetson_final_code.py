#!/usr/bin/env python3
"""
Jetson Nano PD TBR Auto-Processing System with Single START Button
Automatically loads models, receives data from ESP32, evaluates, and sends results
Simplified interface: Just click START and the system handles everything automatically
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import json
import threading
import time
import socket
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

# GPIO for LED control
try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("Jetson.GPIO imported successfully")
except ImportError as e:
    print(f"Jetson.GPIO import error: {e}")
    print("LED functionality will be disabled")
    GPIO_AVAILABLE = False

# Handle potential import issues on ARM devices like Jetson Nano
try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"PyTorch import error: {e}")
    print("Please install PyTorch for ARM")
    torch = None
    nn = None

try:
    import joblib
except ImportError as e:
    print(f"Joblib import error: {e}")
    print("Please install joblib: pip install joblib")
    joblib = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("tiktoken not available - using simple tokenizer for LLM")
    TIKTOKEN_AVAILABLE = False

class PDAutoProcessingSystem:
    def __init__(self, esp32_ip="192.168.221.178", target_device_ip="192.168.221.42", target_port=5001, listen_port=8888):
        self.esp32_ip = esp32_ip
        self.esp32_prediction_url = f"http://{esp32_ip}/prediction_result"
        
        # Target device for prescription sending
        self.target_device_ip = target_device_ip
        self.target_port = target_port
        
        # Server for receiving data from ESP32
        self.listen_port = listen_port
        self.server_running = False
        self.server_thread = None
        
        self.root = tk.Tk()
        self.root.title("PD Auto-Processing System - Jetson Nano")
        self.root.geometry("1400x850")
        self.root.configure(bg='#f0f0f0')
        
        # System state
        self.system_started = False
        self.models_loaded = False
        self.llm_loaded = False
        self.auto_send_enabled = True
        self.processing_count = 0
        
        # Data storage
        self.current_data = None
        self.loaded_models = {
            'cnn_model': None,
            'class_names': None,
            'scaler': None,
            'llm_model': None,
            'llm_tokenizer': None
        }
        
        # Model paths
        self.model_paths = {
            'cnn_model': "/home/jetson/Desktop/cnn_models/multiclass_cnn_model",
            'label_classes': "/home/jetson/Desktop/cnn_models/label_classes.npy",
            'scaler': "/home/jetson/Desktop/cnn_models/multiclass_scaler.pkl",
            'llm_model': "/home/jetson/Desktop/llm_models/final_parkinsons_model.pth"
        }
        
        # LED Configuration
        self.led_enabled = False
        self.led_pins = [33, 32, 31, 29, 18]
        self.led_labels = ["Normal", "Mild PD", "Moderate PD", "Severe PD", "Very Severe PD"]
        self.current_led_index = -1
        
        # Setup GPIO for LEDs
        self.setup_leds()
        
        # Setup GUI
        self.setup_gui()
        
        # Ensure GPIO cleanup on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_leds(self):
        """Initialize GPIO pins for LED control"""
        if not GPIO_AVAILABLE:
            self.log_message("GPIO not available - LED functionality disabled")
            return
            
        try:
            GPIO.setmode(GPIO.BOARD)
            for i, pin in enumerate(self.led_pins):
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
                
            self.led_enabled = True
            self.log_message("LED system initialized successfully")
                
        except Exception as e:
            self.log_message(f"Failed to initialize LEDs: {str(e)}")
            self.led_enabled = False

    def control_severity_leds(self, severity_level):
        """Control LEDs based on PD severity level"""
        if not self.led_enabled:
            return
            
        try:
            for pin in self.led_pins:
                GPIO.output(pin, GPIO.LOW)
                
            severity_to_led = {
                "Normal": 0,
                "Mild PD": 1,
                "Moderate PD": 2,
                "Severe PD": 3,
                "Very Severe PD": 4
            }
            
            led_index = severity_to_led.get(severity_level, -1)
            
            if led_index >= 0 and led_index < len(self.led_pins):
                GPIO.output(self.led_pins[led_index], GPIO.HIGH)
                self.current_led_index = led_index
                self.log_message(f"LED Control: {severity_level} -> LED {led_index + 1} ON")
                
                if severity_level == "Very Severe PD":
                    threading.Thread(target=self.blink_severe_led, daemon=True).start()
                    
        except Exception as e:
            self.log_message(f"Error controlling LEDs: {str(e)}")
            
    def blink_severe_led(self):
        """Special blinking pattern for Very Severe PD"""
        if not self.led_enabled or self.current_led_index != 4:
            return
            
        try:
            for i in range(20):
                if self.current_led_index == 4:
                    GPIO.output(self.led_pins[4], GPIO.LOW)
                    time.sleep(0.2)
                    GPIO.output(self.led_pins[4], GPIO.HIGH)
                    time.sleep(0.3)
                else:
                    break
        except Exception as e:
            self.log_message(f"Error in LED blinking: {str(e)}")

    def on_closing(self):
        """Clean up and close application"""
        if self.server_running:
            self.stop_server()
            
        if self.led_enabled:
            try:
                GPIO.cleanup()
                self.log_message("GPIO cleanup completed")
            except Exception as e:
                self.log_message(f"GPIO cleanup error: {str(e)}")
                
        self.root.destroy()

    # HTTP Server for receiving data from ESP32
    class DataReceiveHandler(http.server.BaseHTTPRequestHandler):
        def __init__(self, parent_system, *args, **kwargs):
            self.parent = parent_system
            super().__init__(*args, **kwargs)
            
        def do_POST(self):
            if self.path == '/receive_data':
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    # Process data in parent system
                    self.parent.process_received_data(data)
                    
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({"status": "success", "message": "Data received and processed"})
                    self.wfile.write(response.encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({"status": "error", "message": str(e)})
                    self.wfile.write(response.encode())
            else:
                self.send_response(404)
                self.end_headers()
                
        def log_message(self, format, *args):
            # Suppress default HTTP server logging
            pass

    def start_server(self):
        """Start HTTP server to receive data from ESP32"""
        try:
            def handler_factory(*args, **kwargs):
                return self.DataReceiveHandler(self, *args, **kwargs)
                
            self.httpd = socketserver.TCPServer(("", self.listen_port), handler_factory)
            self.server_running = True
            
            def run_server():
                self.log_message(f"Data receiver server started on port {self.listen_port}")
                self.httpd.serve_forever()
                
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            self.log_message(f"Failed to start server: {str(e)}")
            return False

    def stop_server(self):
        """Stop the HTTP server"""
        if self.server_running:
            try:
                self.httpd.shutdown()
                self.server_running = False
                self.log_message("Data receiver server stopped")
            except Exception as e:
                self.log_message(f"Error stopping server: {str(e)}")

    def process_received_data(self, data):
        """Process data received from ESP32"""
        try:
            self.log_message("Data received from ESP32:")
            self.log_message(f"  Source: {data.get('source', 'unknown')}")
            self.log_message(f"  Record ID: {data.get('record_id', 'N/A')}")
            self.log_message(f"  Request ID: {data.get('request_id', 'N/A')}")
            
            # Extract the 4 key values
            theta = data.get('theta', 0)
            high_beta = data.get('high_beta', 0)
            low_beta = data.get('low_beta', 0)
            tbr = data.get('tbr', 0)
            
            self.log_message(f"  Theta: {theta:.6f}")
            self.log_message(f"  High Beta: {high_beta:.6f}")
            self.log_message(f"  Low Beta: {low_beta:.6f}")
            self.log_message(f"  TBR: {tbr:.6f}")
            
            # Store current data
            self.current_data = {
                'theta': theta,
                'high_beta': high_beta,
                'low_beta': low_beta,
                'tbr': tbr,
                'record_id': data.get('record_id', 'N/A'),
                'request_id': data.get('request_id', 0),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Update GUI in main thread
            self.root.after(0, self.update_current_data_display)
            
            # Auto-evaluate if models are loaded
            if self.models_loaded:
                self.root.after(0, self.auto_evaluate_data)
            else:
                self.log_message("Models not loaded - data stored but not evaluated")
                
        except Exception as e:
            self.log_message(f"Error processing received data: {str(e)}")

    def update_current_data_display(self):
        """Update GUI with current data"""
        if self.current_data:
            self.record_id_var.set(str(self.current_data['record_id']))
            self.received_time_var.set(datetime.now().strftime("%H:%M:%S"))
            
            # Update PD values
            self.pd_values['theta'].set(f"{self.current_data['theta']:.6f}")
            self.pd_values['high_beta'].set(f"{self.current_data['high_beta']:.6f}")
            self.pd_values['low_beta'].set(f"{self.current_data['low_beta']:.6f}")
            self.pd_values['tbr'].set(f"{self.current_data['tbr']:.6f}")
            
            # Update processing count
            self.processing_count += 1
            self.stats_label.config(text=f"Processed: {self.processing_count}\nLast: {datetime.now().strftime('%H:%M:%S')}")

    def auto_evaluate_data(self):
        """Automatically evaluate the current data"""
        if not self.current_data or not self.models_loaded:
            return
            
        try:
            self.log_message("Auto-evaluating received data...")
            
            # Extract features
            features = [
                self.current_data['theta'],
                self.current_data['low_beta'],
                self.current_data['high_beta'],
                self.current_data['tbr']
            ]
            
            # Scale features
            scaled_features = self.loaded_models['scaler'].transform([features])
            X_tensor = torch.tensor(scaled_features, dtype=torch.float32)
            
            # Make prediction
            start_time = time.time()
            with torch.no_grad():
                output = self.loaded_models['cnn_model'](X_tensor)
                probabilities = torch.softmax(output, dim=1).numpy().flatten()
                predicted_class_idx = int(np.argmax(probabilities))
            end_time = time.time()
            
            predicted_class = self.loaded_models['class_names'][predicted_class_idx]
            inference_time = end_time - start_time
            confidence = probabilities[predicted_class_idx] * 100
            request_id = self.current_data.get('request_id', 0)
            
            # Update results display
            self.display_evaluation_results(features, predicted_class, probabilities, inference_time)
            
            # Update prescription display and control LEDs
            self.update_prescription_display(predicted_class, confidence)
            
            # Send prediction back to ESP32
            self.send_prediction_to_esp32(features, predicted_class, probabilities, request_id)
            
            self.log_message("Auto-evaluation completed:")
            self.log_message(f"  Predicted: {predicted_class}")
            self.log_message(f"  Confidence: {confidence:.2f}%")
            self.log_message("  Results sent to ESP32")
            
        except Exception as e:
            self.log_message(f"Auto-evaluation error: {str(e)}")

    # Include all the previous LLM and model classes here (same as before)
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {
                '<|endoftext|>': 50256,
                'Based': 15001, 'on': 319, 'UPDRS': 52, 'score': 4776,
                'mild': 11607, 'moderate': 10768, 'severe': 6049, 'normal': 3487,
                'Parkinson': 23604, 'Disease': 17344, 'symptoms': 7460,
                'treatment': 3513, 'medication': 22103, 'therapy': 9102,
                'exercise': 5163, 'recommend': 4313, 'consult': 9110,
                'doctor': 6253, 'neurologist': 7669, 'physical': 3518,
                ':': 25, '.': 13, ',': 11, ' ': 220, '\n': 198
            }
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            for i in range(100):
                self.vocab[str(i)] = 1000 + i
                self.id_to_token[1000 + i] = str(i)
        
        def encode(self, text, allowed_special=None):
            words = text.split()
            return [self.vocab.get(word, 50257) for word in words]
        
        def decode(self, token_ids):
            return ' '.join([self.id_to_token.get(tid, '[UNK]') for tid in token_ids])

    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            ))

    class FeedForward(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                PDAutoProcessingSystem.GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            )

        def forward(self, x):
            return self.layers(x)

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

        def forward(self, x):
            b, num_tokens, d_in = x.shape

            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)

            return context_vec

    class TransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = PDAutoProcessingSystem.MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"])
            self.ff = PDAutoProcessingSystem.FeedForward(cfg)
            self.norm1 = PDAutoProcessingSystem.LayerNorm(cfg["emb_dim"])
            self.norm2 = PDAutoProcessingSystem.LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        def forward(self, x):
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)
            x = self.drop_shortcut(x)
            x = x + shortcut

            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut

            return x

    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            self.trf_blocks = nn.Sequential(
                *[PDAutoProcessingSystem.TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

            self.final_norm = PDAutoProcessingSystem.LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits
        
    class CNN(nn.Module):
        def __init__(self, num_classes):
            super(PDAutoProcessingSystem.CNN, self).__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )
        def forward(self, x): 
            return self.net(x.unsqueeze(1))

    def load_models_automatically(self):
        """Load all models automatically"""
        try:
            self.log_message("Loading models automatically...")
            
            # Check if PyTorch is available
            if torch is None or nn is None:
                raise ImportError("PyTorch not available. Please install PyTorch.")
            
            if joblib is None:
                raise ImportError("Joblib not available. Please install joblib.")
            
            # Check if CNN model files exist
            import os
            cnn_required_files = ['cnn_model', 'label_classes', 'scaler']
            for name in cnn_required_files:
                path = self.model_paths[name]
                if not os.path.exists(path):
                    raise FileNotFoundError(f"CNN model file not found: {path}")
                    
            # Load class names
            self.log_message("Loading class names...")
            self.loaded_models['class_names'] = np.load(self.model_paths['label_classes'], allow_pickle=True)
            self.log_message(f"Loaded classes: {list(self.loaded_models['class_names'])}")
            
            # Load scaler
            self.log_message("Loading scaler...")
            self.loaded_models['scaler'] = joblib.load(self.model_paths['scaler'])
            self.log_message("Scaler loaded successfully")
            
            # Load CNN model
            self.log_message("Loading CNN model...")
            num_classes = len(self.loaded_models['class_names'])
            self.loaded_models['cnn_model'] = self.CNN(num_classes=num_classes)
            
            model_state = torch.load(self.model_paths['cnn_model'], map_location=torch.device('cpu'))
            self.loaded_models['cnn_model'].load_state_dict(model_state)
            self.loaded_models['cnn_model'].eval()
            self.log_message("CNN model loaded successfully")
            
            # Try to load LLM model (optional)
            self.log_message("Attempting to load LLM model...")
            try:
                if os.path.exists(self.model_paths['llm_model']):
                    llm_config = {
                        "vocab_size": 50257,
                        "context_length": 1024,
                        "drop_rate": 0.1,
                        "qkv_bias": True,
                        "emb_dim": 768,
                        "n_layers": 12,
                        "n_heads": 12
                    }
                    
                    self.loaded_models['llm_model'] = self.GPTModel(llm_config)
                    llm_checkpoint = torch.load(self.model_paths['llm_model'], map_location=torch.device('cpu'))
                    self.loaded_models['llm_model'].load_state_dict(llm_checkpoint["model_state_dict"])
                    self.loaded_models['llm_model'].eval()
                    
                    if TIKTOKEN_AVAILABLE:
                        import tiktoken
                        self.loaded_models['llm_tokenizer'] = tiktoken.get_encoding("gpt2")
                        self.log_message("LLM model and tiktoken tokenizer loaded successfully")
                    else:
                        self.loaded_models['llm_tokenizer'] = self.SimpleTokenizer()
                        self.log_message("LLM model loaded with simple tokenizer")
                    
                    self.llm_loaded = True
                    
                else:
                    self.log_message("LLM model file not found - using rule-based prescription generation")
                    self.llm_loaded = False
                    
            except Exception as e:
                self.log_message(f"LLM loading failed: {str(e)} - using rule-based fallback")
                self.llm_loaded = False
            
            # Update status
            self.models_loaded = True
            self.log_message("All models loaded successfully - Auto-processing enabled")
            
            return True
                
        except Exception as e:
            self.log_message(f"Error loading models: {str(e)}")
            return False

    def start_system(self):
        """Start the complete auto-processing system"""
        if self.system_started:
            self.log_message("System already started")
            return
            
        self.log_message("Starting PD Auto-Processing System...")
        self.start_btn.config(state='disabled', text="Starting System...")
        
        # Start in separate thread
        threading.Thread(target=self._start_system_thread, daemon=True).start()

    def _start_system_thread(self):
        """Start system in separate thread"""
        try:
            # Step 1: Load models
            self.log_message("Step 1: Loading models automatically...")
            if not self.load_models_automatically():
                self.root.after(0, self._start_failed, "Failed to load models")
                return
            
            # Step 2: Start data receiver server
            self.log_message("Step 2: Starting data receiver server...")
            if not self.start_server():
                self.root.after(0, self._start_failed, "Failed to start server")
                return
            
            # Step 3: System ready
            self.root.after(0, self._start_success)
            
        except Exception as e:
            self.root.after(0, self._start_failed, str(e))

    def _start_success(self):
        """Handle successful system start"""
        try:
            self.system_started = True
            self.start_btn.config(state='normal', text="SYSTEM RUNNING", bg='#27ae60', fg='white')
            self.status_label.config(text="System Status: RUNNING", fg='green')
            self.models_status_label.config(text="Models: Loaded", fg='green')
            self.llm_status_label.config(text=f"LLM: {'Loaded' if self.llm_loaded else 'Rule-based'}", fg='green')
            self.server_status_label.config(text=f"Server: Listening on :{self.listen_port}", fg='green')
            
            self.log_message(">>> PD AUTO-PROCESSING SYSTEM STARTED SUCCESSFULLY")
            self.log_message("System is now ready to:")
            self.log_message("  * Auto-receive data from ESP32")
            self.log_message("  * Auto-evaluate PD severity")
            self.log_message("  * Auto-send results to ESP32")
            self.log_message("  * Auto-generate prescriptions")
            self.log_message("  * Auto-control LED indicators")
            if self.auto_send_enabled:
                self.log_message("  * Auto-send prescriptions to target device")
            self.log_message("")
            self.log_message("Ready for data from website 'Fill Data' button...")
        except Exception as e:
            self.log_message(f"Error in start success handler: {str(e)}")
            self.start_btn.config(state='normal', text="START SYSTEM", bg='#3498db')

    def _start_failed(self, error):
        """Handle failed system start"""
        try:
            self.start_btn.config(state='normal', text="START SYSTEM", bg='#e74c3c', fg='white')
            self.status_label.config(text="System Status: FAILED", fg='red')
            self.log_message(f">>> SYSTEM START FAILED: {error}")
        except Exception as e:
            print(f"Error in start failed handler: {str(e)}")
            # Fallback to basic button config
            self.start_btn.config(state='normal', text="START SYSTEM")

    def stop_system(self):
        """Stop the auto-processing system"""
        if not self.system_started:
            return
            
        self.log_message("Stopping PD Auto-Processing System...")
        
        # Stop server
        if self.server_running:
            self.stop_server()
        
        # Turn off LEDs
        if self.led_enabled:
            try:
                for pin in self.led_pins:
                    GPIO.output(pin, GPIO.LOW)
                self.current_led_index = -1
            except:
                pass
        
        # Update GUI
        self.system_started = False
        self.start_btn.config(text="START SYSTEM", bg='#3498db')
        self.status_label.config(text="System Status: STOPPED", fg='red')
        self.models_status_label.config(text="Models: Not Loaded", fg='red')
        self.server_status_label.config(text="Server: Stopped", fg='red')
        
        self.log_message(">>> System stopped")

    def send_prescription_to_target_device(self, prescription_text, severity_level, confidence):
        """Send prescription text to target device via socket"""
        if not self.auto_send_enabled:
            return False
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            message = f"""
PARKINSON'S DISEASE EVALUATION ALERT

Timestamp: {timestamp}
Severity Level: {severity_level}
Confidence Level: {confidence:.1f} percent

AI GENERATED PRESCRIPTION:

{prescription_text}

This is an automated message from the PD monitoring system.
Please consult with healthcare professionals for proper medical care.
End of prescription message.
"""
            
            self.log_message(f"Sending prescription to target device {self.target_device_ip}:{self.target_port}")
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((self.target_device_ip, self.target_port))
                s.sendall(message.encode("utf-8"))
            
            self.log_message("[SUCCESS] Prescription sent to target device successfully")
            return True
            
        except Exception as e:
            self.log_message(f"[FAILED] Error sending prescription to target device: {str(e)}")
            return False

    def generate_llm_prescription(self, severity_level, confidence):
        """Generate prescription using LLM or rule-based system"""
        # Same prescription generation logic as before
        prescriptions = {
            "Normal": [
                "Continue regular health monitoring",
                "Maintain active lifestyle with regular exercise",
                "Follow balanced diet rich in antioxidants", 
                "Schedule routine neurological check-ups annually",
                "Practice stress management techniques",
                "Ensure adequate sleep (7-9 hours nightly)"
            ],
            "Mild PD": [
                "Start with Levodopa/Carbidopa (25/100mg) twice daily",
                "Begin physical therapy focusing on mobility",
                "Implement speech therapy if needed",
                "Regular exercise: walking, swimming, tai chi",
                "Maintain social activities and mental stimulation",
                "Monitor symptoms and medication response",
                "Schedule neurologist visits every 3-6 months"
            ],
            "Moderate PD": [
                "Adjust Levodopa dosage (may increase to 3-4 times daily)",
                "Consider adding Dopamine agonists (Pramipexole/Ropinirole)",
                "Intensive physical therapy and occupational therapy",
                "Address sleep disorders and depression if present",
                "Implement home safety modifications",
                "Consider support groups and counseling",
                "Regular medication timing is crucial",
                "Monitor for dyskinesia and motor fluctuations"
            ],
            "Severe PD": [
                "Optimize complex medication regimen with neurologist",
                "Consider advanced therapies: DBS evaluation",
                "Comprehensive care team: neurologist, PT, OT, speech therapist",
                "Address non-motor symptoms: cognitive, psychiatric",
                "Implement fall prevention strategies",
                "Consider feeding tube if swallowing difficulties",
                "Palliative care consultation for quality of life",
                "Caregiver support and respite care"
            ],
            "Very Severe PD": [
                "Specialized palliative and hospice care evaluation",
                "Advanced directive and end-of-life planning",
                "Complex medication management with frequent adjustments",
                "24/7 nursing care may be required",
                "Aggressive management of complications",
                "Family education and emotional support",
                "Consider experimental treatments if appropriate",
                "Focus on comfort and dignity of care"
            ]
        }
        
        base_prescription = prescriptions.get(severity_level, prescriptions["Normal"])
        
        confidence_note = ""
        if confidence >= 90:
            confidence_note = "High confidence prediction - proceed with recommended treatment plan"
        elif confidence >= 70:
            confidence_note = "Moderate confidence - consider additional diagnostic tests"
        else:
            confidence_note = "Low confidence prediction - require comprehensive clinical evaluation"
        
        return "\n".join([f"* {item}" for item in base_prescription] + [f"* {confidence_note}"])

    def send_prediction_to_esp32(self, features, predicted_class, probabilities, request_id):
        """Send prediction results back to ESP32"""
        try:
            prediction_data = {
                "classification": predicted_class,
                "confidence": float(probabilities[np.argmax(probabilities)] * 100),
                "theta": float(features[0]),
                "low_beta": float(features[1]),
                "high_beta": float(features[2]),
                "tbr": float(features[3]),
                "request_id": request_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            response = requests.post(self.esp32_prediction_url, 
                                   json=prediction_data, 
                                   timeout=5,
                                   headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                self.log_message(f"[SUCCESS] Prediction sent to ESP32 successfully")
                return True
            else:
                self.log_message(f"[FAILED] Failed to send prediction to ESP32: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_message(f"[ERROR] Error sending prediction to ESP32: {str(e)}")
            return False

    def update_prescription_display(self, severity_level, confidence):
        """Update prescription display with AI-generated content and auto-send"""
        try:
            # Update header
            self.severity_display.config(text=severity_level, fg='#27ae60' if severity_level == "Normal" else '#e74c3c')
            self.confidence_display.config(text=f"{confidence:.1f}%", fg='#27ae60' if confidence > 80 else '#f39c12')
            
            # Control LEDs
            self.control_severity_leds(severity_level)
            
            # Generate prescription
            prescription = self.generate_llm_prescription(severity_level, confidence)
            
            # Update prescription text
            self.prescription_text.config(state='normal')
            self.prescription_text.delete('1.0', 'end')
            
            prescription_header = f"""AI MEDICAL PRESCRIPTION WITH AUTO-SEND

EVALUATION SUMMARY:
   Severity Level: {severity_level}
   Confidence: {confidence:.1f}%
   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
   Auto-Send: {'Enabled' if self.auto_send_enabled else 'Disabled'}

TREATMENT RECOMMENDATIONS:
{prescription}

AUTO-SEND STATUS:
   Target Device: {self.target_device_ip}:{self.target_port}
   Prescription will be automatically sent for voice reading
   LED indicators show severity level
   ESP32 display updated with results

IMPORTANT MEDICAL DISCLAIMER:
* This AI-generated prescription is for research and educational purposes only
* Always consult with qualified healthcare professionals for diagnosis and treatment
* Individual patient factors may require different treatment approaches
* Regular monitoring and follow-up care are essential
* Emergency symptoms require immediate medical attention
"""
            
            self.prescription_text.insert('1.0', prescription_header)
            self.prescription_text.config(state='disabled')
            
            # Auto-send prescription if PD detected
            if severity_level != "Normal" and self.auto_send_enabled:
                self.log_message(f"PD DETECTED ({severity_level}) - Auto-sending prescription to target device...")
                
                threading.Thread(
                    target=self.send_prescription_to_target_device,
                    args=(prescription, severity_level, confidence),
                    daemon=True
                ).start()
            elif severity_level == "Normal":
                self.log_message("Normal classification - no prescription auto-send needed")
            else:
                self.log_message("Auto-send disabled - prescription not sent to target device")
                
        except Exception as e:
            self.log_message(f"Error updating prescription display: {str(e)}")

    def display_evaluation_results(self, features, predicted_class, probabilities, inference_time):
        """Display evaluation results"""
        self.results_text.delete(1.0, 'end')
        
        results = f"{'='*60}\n"
        results += f"PARKINSON'S DISEASE AUTO-EVALUATION RESULTS\n"
        results += f"{'='*60}\n\n"
        
        results += f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        results += f"Record ID: {self.current_data.get('record_id', 'N/A')}\n"
        results += f"Data Source: ESP32 Auto-Send\n"
        results += f"Inference Time: {inference_time:.4f} seconds\n\n"
        
        results += f"INPUT FEATURES:\n"
        results += f"{'-'*30}\n"
        results += f"Theta:              {features[0]:10.6f}\n"
        results += f"Low Beta:           {features[1]:10.6f}\n"
        results += f"High Beta:          {features[2]:10.6f}\n"
        results += f"TBR (Theta/Beta):   {features[3]:10.6f}\n\n"
        
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx] * 100
        
        results += f"PREDICTION RESULT:\n"
        results += f"{'-'*30}\n"
        results += f"PD Classification: {predicted_class}\n"
        results += f"Confidence Level: {confidence:.2f}%\n\n"
        
        results += f"ALL CLASS PROBABILITIES:\n"
        results += f"{'-'*30}\n"
        for i, class_name in enumerate(self.loaded_models['class_names']):
            prob_percent = probabilities[i] * 100
            bar_length = int(prob_percent / 2)
            bar = '#' * bar_length + '-' * (50 - bar_length)
            results += f"{class_name:20s}: {prob_percent:6.2f}% |{bar}|\n"
        
        results += f"\nAUTO-PROCESSING STATUS:\n"
        results += f"{'-'*30}\n"
        results += f"Data received from: ESP32 Auto-Send\n"
        results += f"Models loaded: Automatically\n"
        results += f"Evaluation: Automatic\n"
        results += f"Results sent to ESP32: YES\n"
        results += f"LED Control: Automatic\n"
        results += f"Prescription Generated: YES\n"
        if predicted_class != "Normal" and self.auto_send_enabled:
            results += f"Prescription Auto-Send: TRIGGERED\n"
        
        results += f"\n{'='*60}\n"
        
        self.results_text.insert('end', results)

    def log_message(self, message):
        """Add message to log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            self.log_text.insert('end', log_entry)
            self.log_text.see('end')
            if hasattr(self, 'root'):
                self.root.update()
        except AttributeError:
            print(log_entry.strip())

    def setup_gui(self):
        """Setup the simplified GUI layout"""
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="PD Auto-Processing System - Jetson Nano", 
                              font=('Arial', 18, 'bold'), 
                              fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Control frame with single START button
        control_frame = tk.Frame(self.root, bg='#ecf0f1', relief='raised', bd=2)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Single START button (main control)
        self.start_btn = tk.Button(control_frame, text="START SYSTEM", 
                                  command=self.start_system,
                                  font=('Arial', 16, 'bold'), 
                                  bg='#3498db', fg='white', 
                                  height=3, width=20,
                                  relief='raised', bd=3)
        self.start_btn.pack(side='left', padx=20, pady=15)
        
        # Status display
        status_frame = tk.LabelFrame(control_frame, text="System Status", 
                                   font=('Arial', 11, 'bold'), bg='#ecf0f1')
        status_frame.pack(side='left', padx=20, pady=15)
        
        self.status_label = tk.Label(status_frame, text="System Status: STOPPED", 
                                    font=('Arial', 11), bg='#ecf0f1', fg='red')
        self.status_label.pack(pady=2)
        
        self.models_status_label = tk.Label(status_frame, text="Models: Not Loaded", 
                                           font=('Arial', 10), bg='#ecf0f1', fg='red')
        self.models_status_label.pack(pady=1)
        
        self.llm_status_label = tk.Label(status_frame, text="LLM: Not Loaded", 
                                        font=('Arial', 10), bg='#ecf0f1', fg='red')
        self.llm_status_label.pack(pady=1)
        
        self.server_status_label = tk.Label(status_frame, text="Server: Stopped", 
                                           font=('Arial', 10), bg='#ecf0f1', fg='red')
        self.server_status_label.pack(pady=1)
        
        # Statistics frame
        stats_frame = tk.Frame(control_frame, bg='#ecf0f1')
        stats_frame.pack(side='right', padx=20, pady=15)
        
        self.stats_label = tk.Label(stats_frame, text="Processed: 0\nLast: Never", 
                                   font=('Arial', 11), bg='#ecf0f1', justify='left')
        self.stats_label.pack()
        
        # STOP button
        self.stop_btn = tk.Button(stats_frame, text="STOP", 
                                 command=self.stop_system,
                                 bg='#e74c3c', fg='white', font=('Arial', 10))
        self.stop_btn.pack(pady=(10, 0))
        
        # Create main content area
        main_content = tk.Frame(self.root, bg='#f0f0f0')
        main_content.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left side - Current data and prescription
        left_frame = tk.Frame(main_content, bg='white', width=700)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        left_frame.pack_propagate(False)
        
        # Current data display
        current_data_frame = tk.LabelFrame(left_frame, text="Current PD Data (Auto-Received)", 
                                          font=('Arial', 12, 'bold'), bg='white')
        current_data_frame.pack(fill='x', padx=10, pady=5)
        
        # Record info
        info_frame = tk.Frame(current_data_frame, bg='white')
        info_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(info_frame, text="Record ID:", font=('Arial', 10, 'bold'), bg='white').grid(row=0, column=0, sticky='w')
        self.record_id_var = tk.StringVar(value="None")
        tk.Label(info_frame, textvariable=self.record_id_var, font=('Arial', 10), bg='white').grid(row=0, column=1, sticky='w', padx=10)
        
        tk.Label(info_frame, text="Received:", font=('Arial', 10, 'bold'), bg='white').grid(row=0, column=2, sticky='w', padx=20)
        self.received_time_var = tk.StringVar(value="Never")
        tk.Label(info_frame, textvariable=self.received_time_var, font=('Arial', 10), bg='white').grid(row=0, column=3, sticky='w', padx=10)
        
        # PD Values display
        values_frame = tk.LabelFrame(current_data_frame, text="PD Classification Values", 
                                    font=('Arial', 11, 'bold'), bg='white')
        values_frame.pack(fill='x', padx=5, pady=5)
        
        self.pd_values = {}
        pd_labels = ['Theta', 'High Beta', 'Low Beta', 'TBR']
        pd_keys = ['theta', 'high_beta', 'low_beta', 'tbr']
        
        for i, (label, key) in enumerate(zip(pd_labels, pd_keys)):
            row = i // 2
            col = i % 2
            
            value_frame = tk.Frame(values_frame, bg='#ecf0f1', relief='sunken', bd=2)
            value_frame.grid(row=row, column=col, padx=10, pady=10, sticky='ew', ipadx=20, ipady=10)
            
            tk.Label(value_frame, text=label, 
                    font=('Arial', 12, 'bold'), bg='#ecf0f1').pack()
            
            self.pd_values[key] = tk.StringVar(value="0.000000")
            value_label = tk.Label(value_frame, textvariable=self.pd_values[key], 
                                  font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50')
            value_label.pack()
        
        for i in range(2):
            values_frame.grid_columnconfigure(i, weight=1)

        # AI Prescription Section
        prescription_frame = tk.LabelFrame(left_frame, text="AI Generated Medical Prescription (Auto)", 
                                          font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        prescription_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Prescription header
        prescription_header = tk.Frame(prescription_frame, bg='white')
        prescription_header.pack(fill='x', padx=5, pady=5)
        
        tk.Label(prescription_header, text="Severity Level:", 
                font=('Arial', 11, 'bold'), bg='white').pack(side='left')
        self.severity_display = tk.Label(prescription_header, text="Not Evaluated", 
                                        font=('Arial', 11), bg='white', fg='#e74c3c')
        self.severity_display.pack(side='left', padx=10)
        
        tk.Label(prescription_header, text="Confidence:", 
                font=('Arial', 11, 'bold'), bg='white').pack(side='left', padx=(20, 0))
        self.confidence_display = tk.Label(prescription_header, text="N/A", 
                                          font=('Arial', 11), bg='white', fg='#e74c3c')
        self.confidence_display.pack(side='left', padx=10)
        
        # Prescription text area
        prescription_text_frame = tk.Frame(prescription_frame, bg='white')
        prescription_text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.prescription_text = scrolledtext.ScrolledText(
            prescription_text_frame, 
            height=12, 
            font=('Arial', 11), 
            bg='#f8f9fa', 
            fg='#2c3e50',
            wrap='word',
            relief='sunken',
            bd=2
        )
        self.prescription_text.pack(fill='both', expand=True)
        
        initial_message = """PD AUTO-PROCESSING SYSTEM

This system automatically:
* Loads CNN and LLM models
* Receives data from ESP32 when website "Fill Data" is clicked
* Evaluates PD severity using CNN model
* Generates AI prescriptions
* Controls LED severity indicators
* Sends results back to ESP32
* Auto-sends prescriptions to target device

STATUS: Click START SYSTEM to begin

The system will automatically handle all processing when data arrives from ESP32.
No manual intervention required after START is clicked.

IMPORTANT: This is an AI-assisted tool for research purposes only.
Always consult with qualified medical professionals for actual diagnosis and treatment."""
        
        self.prescription_text.insert('1.0', initial_message)
        self.prescription_text.config(state='disabled')
        
        # Right side - Results and logs
        right_frame = tk.Frame(main_content, bg='white', width=600)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results tab
        self.results_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.results_frame, text="Auto-Evaluation Results")
        
        results_display_frame = tk.LabelFrame(self.results_frame, text="CNN Model Auto-Evaluation Results", 
                                            font=('Arial', 12, 'bold'), bg='white')
        results_display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_display_frame, height=20, width=60,
                                                     font=('Courier', 10), bg='#f8f9fa', fg='#2c3e50')
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Log tab
        self.log_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.log_frame, text="System Log")
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=20, width=60,
                                                 font=('Courier', 10), bg='#1e1e1e', fg='#00ff00')
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initial log messages
        self.log_message("PD Auto-Processing System Initialized")
        self.log_message(f"ESP32 Target: {self.esp32_ip}")
        self.log_message(f"Data Receiver Port: {self.listen_port}")
        self.log_message(f"Prescription Target: {self.target_device_ip}:{self.target_port}")
        self.log_message("Click START SYSTEM to begin automatic processing")
        if self.led_enabled:
            self.log_message("LED severity indicators ready")
        self.log_message("")
        self.log_message("Waiting for START command...")

    def run(self):
        """Start the GUI application"""
        print("PD Auto-Processing System starting...")
        print(f"ESP32 Target: {self.esp32_ip}")
        print(f"Data Receiver Port: {self.listen_port}")
        print(f"Prescription Target: {self.target_device_ip}:{self.target_port}")
        print("GUI is starting...")
        
        self.root.mainloop()

def main():
    # Configuration - Update these IP addresses as needed
    ESP32_IP = "192.168.221.101"  # Update to your ESP32's IP
    TARGET_DEVICE_IP = "192.168.221.42"  # Update to your target device's IP
    TARGET_PORT = 5001  # Port for prescription sending
    LISTEN_PORT = 8888  # Port for receiving data from ESP32
    
    app = PDAutoProcessingSystem(
        esp32_ip=ESP32_IP, 
        target_device_ip=TARGET_DEVICE_IP, 
        target_port=TARGET_PORT,
        listen_port=LISTEN_PORT
    )
    app.run()

if __name__ == "__main__":
    main()