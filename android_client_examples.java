// Android API Client Classes for Device Performance Model Integration

// 1. Data Models
public class DeviceSpecs {
    public int num_cores;
    public double processor_speed;
    public int battery_capacity;
    public int fast_charging_available;
    public int ram_capacity;
    public int internal_memory;
    public double screen_size;
    public int refresh_rate;
    public String os;
    public int resolution_height;
    public int resolution_width;
    
    public DeviceSpecs(int cores, double speed, int battery, int fastCharging, 
                      int ram, int storage, double screenSize, int refreshRate, 
                      String os, int resHeight, int resWidth) {
        this.num_cores = cores;
        this.processor_speed = speed;
        this.battery_capacity = battery;
        this.fast_charging_available = fastCharging;
        this.ram_capacity = ram;
        this.internal_memory = storage;
        this.screen_size = screenSize;
        this.refresh_rate = refreshRate;
        this.os = os;
        this.resolution_height = resHeight;
        this.resolution_width = resWidth;
    }
}

public class PredictionResult {
    public String status;
    public String category;
    public double confidence;
    public double performance_score;
    public Map<String, Double> probabilities;
    public String timestamp;
}

// 2. API Service Interface
public interface DevicePerformanceAPI {
    @POST("/predict")
    Call<PredictionResult> predictPerformance(@Body DeviceSpecs deviceSpecs);
    
    @POST("/batch_predict")
    Call<BatchPredictionResult> batchPredict(@Body BatchRequest request);
    
    @GET("/health")
    Call<HealthResponse> healthCheck();
    
    @GET("/model_info")
    Call<ModelInfo> getModelInfo();
}

// 3. API Client Class
public class DevicePerformanceClient {
    private static final String BASE_URL = "http://your-server-ip:5000/";
    private DevicePerformanceAPI apiService;
    
    public DevicePerformanceClient() {
        OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .addInterceptor(new HttpLoggingInterceptor().setLevel(HttpLoggingInterceptor.Level.BODY))
            .build();
            
        Retrofit retrofit = new Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build();
            
        apiService = retrofit.create(DevicePerformanceAPI.class);
    }
    
    public void predictDevicePerformance(DeviceSpecs specs, Callback<PredictionResult> callback) {
        Call<PredictionResult> call = apiService.predictPerformance(specs);
        call.enqueue(callback);
    }
    
    public void checkHealth(Callback<HealthResponse> callback) {
        Call<HealthResponse> call = apiService.healthCheck();
        call.enqueue(callback);
    }
}

// 4. Usage Example in Activity/Fragment
public class MainActivity extends AppCompatActivity {
    private DevicePerformanceClient client;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        client = new DevicePerformanceClient();
        
        // Example device specs
        DeviceSpecs deviceSpecs = new DeviceSpecs(
            8,      // num_cores
            2.8,    // processor_speed
            4000,   // battery_capacity
            1,      // fast_charging_available
            8,      // ram_capacity
            128,    // internal_memory
            6.1,    // screen_size
            90,     // refresh_rate
            "android", // os
            2400,   // resolution_height
            1080    // resolution_width
        );
        
        predictPerformance(deviceSpecs);
    }
    
    private void predictPerformance(DeviceSpecs specs) {
        client.predictDevicePerformance(specs, new Callback<PredictionResult>() {
            @Override
            public void onResponse(Call<PredictionResult> call, Response<PredictionResult> response) {
                if (response.isSuccessful() && response.body() != null) {
                    PredictionResult result = response.body();
                    
                    // Update UI with results
                    runOnUiThread(() -> {
                        updateUI(result);
                    });
                } else {
                    Log.e("API", "Prediction failed: " + response.message());
                }
            }
            
            @Override
            public void onFailure(Call<PredictionResult> call, Throwable t) {
                Log.e("API", "Network error: " + t.getMessage());
                // Handle network error
            }
        });
    }
    
    private void updateUI(PredictionResult result) {
        // Update your UI elements
        TextView categoryText = findViewById(R.id.category_text);
        TextView scoreText = findViewById(R.id.score_text);
        TextView confidenceText = findViewById(R.id.confidence_text);
        
        categoryText.setText("Performance: " + result.category);
        scoreText.setText("Score: " + String.format("%.1f", result.performance_score));
        confidenceText.setText("Confidence: " + String.format("%.2f", result.confidence * 100) + "%");
        
        // Update performance probabilities
        for (Map.Entry<String, Double> entry : result.probabilities.entrySet()) {
            Log.d("Performance", entry.getKey() + ": " + entry.getValue());
        }
    }
}

// 5. Device Info Helper Class
public class DeviceInfoHelper {
    
    public static DeviceSpecs getCurrentDeviceSpecs(Context context) {
        // Get device specifications from Android system
        
        // CPU info
        int numCores = Runtime.getRuntime().availableProcessors();
        double processorSpeed = 2.0; // Default, harder to get programmatically
        
        // RAM info
        ActivityManager activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memoryInfo);
        int ramCapacity = (int) (memoryInfo.totalMem / (1024 * 1024 * 1024)); // GB
        
        // Storage info
        StatFs stat = new StatFs(Environment.getDataDirectory().getPath());
        long totalBytes = stat.getBlockCountLong() * stat.getBlockSizeLong();
        int storage = (int) (totalBytes / (1024 * 1024 * 1024)); // GB
        
        // Screen info
        DisplayMetrics metrics = context.getResources().getDisplayMetrics();
        double screenSize = getScreenSizeInInches(metrics);
        int resolutionHeight = metrics.heightPixels;
        int resolutionWidth = metrics.widthPixels;
        
        // Battery info
        BatteryManager batteryManager = (BatteryManager) context.getSystemService(Context.BATTERY_SERVICE);
        int batteryCapacity = getBatteryCapacity(batteryManager);
        
        // Default values for harder to detect specs
        int refreshRate = 60; // Default
        int fastCharging = 0; // Default
        String os = "android";
        
        return new DeviceSpecs(
            numCores, processorSpeed, batteryCapacity, fastCharging,
            ramCapacity, storage, screenSize, refreshRate, os,
            resolutionHeight, resolutionWidth
        );
    }
    
    private static double getScreenSizeInInches(DisplayMetrics metrics) {
        double widthInches = metrics.widthPixels / metrics.xdpi;
        double heightInches = metrics.heightPixels / metrics.ydpi;
        return Math.sqrt(widthInches * widthInches + heightInches * heightInches);
    }
    
    private static int getBatteryCapacity(BatteryManager batteryManager) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            return batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
        }
        return 4000; // Default fallback
    }
}

// 6. Permissions needed in AndroidManifest.xml
/*
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
*/

// 7. Dependencies for build.gradle (Module: app)
/*
dependencies {
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.9.3'
    implementation 'com.google.code.gson:gson:2.10.1'
}
*/
