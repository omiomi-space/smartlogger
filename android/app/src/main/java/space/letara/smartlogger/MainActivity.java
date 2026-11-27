package space.letara.smartlogger;

import android.Manifest;
import android.app.Activity;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Streams IMU (rotation vector, accelerometer, gyroscope, magnetometer, linear acceleration),
 * barometric pressure and GPS as JSON over UDP. Packet format is documented in the repo README.
 */
public class MainActivity extends Activity implements SensorEventListener, LocationListener {

    private static final int REQ_LOCATION = 1;
    private static final int REQ_CAMERA = 2;

    private SensorManager sensorManager;
    private LocationManager locationManager;

    private EditText hostEdit, portEdit, rateEdit;
    private Button startButton;
    private TextView statusView;
    private CheckBox arCheck;
    private ArTracker ar;
    private boolean arActive = false;

    private final Handler ui = new Handler(Looper.getMainLooper());
    private ScheduledExecutorService sender;
    private volatile DatagramSocket socket;
    private volatile InetSocketAddress target;      // configured destination
    private volatile InetSocketAddress helloTarget; // PC that sent us a hello (takes priority)
    private volatile long lastHelloMs = 0;
    private Thread helloThread;
    private static final String HELLO = "SMARTLOGGER_HELLO";
    private static final long HELLO_TIMEOUT_MS = 5000;
    private volatile boolean running = false;
    private long seq = 0;
    private volatile String lastError = null;

    // Latest sensor values (guarded by "this").
    private final float[] quat = {1, 0, 0, 0};
    private final float[] acc = new float[3];
    private final float[] gyr = new float[3];
    private final float[] mag = new float[3];
    private final float[] lin = new float[3];
    private float pressure = Float.NaN;
    private boolean hasQuat, hasAcc, hasGyr, hasMag, hasLin;
    private String quatSource = "none";
    private volatile Location lastLocation;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        locationManager = (LocationManager) getSystemService(LOCATION_SERVICE);
        buildUi();
        ui.post(statusTick);
        if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION}, REQ_LOCATION);
        }
    }

    private void buildUi() {
        SharedPreferences prefs = getPreferences(MODE_PRIVATE);
        int pad = dp(16);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("SmartLogger UDP Sender");
        title.setTextSize(22);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);

        root.addView(label("PC IP address (255.255.255.255 = broadcast)"));
        hostEdit = new EditText(this);
        hostEdit.setSingleLine();
        hostEdit.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        hostEdit.setText(prefs.getString("host", "255.255.255.255"));
        root.addView(hostEdit);

        root.addView(label("UDP port"));
        portEdit = new EditText(this);
        portEdit.setInputType(InputType.TYPE_CLASS_NUMBER);
        portEdit.setText(String.valueOf(prefs.getInt("port", 5005)));
        root.addView(portEdit);

        root.addView(label("Send rate (Hz, 1-200)"));
        rateEdit = new EditText(this);
        rateEdit.setInputType(InputType.TYPE_CLASS_NUMBER);
        rateEdit.setText(String.valueOf(prefs.getInt("rate", 50)));
        root.addView(rateEdit);

        startButton = new Button(this);
        startButton.setText("START");
        startButton.setOnClickListener(v -> {
            if (running) stopStreaming(); else startStreaming();
        });
        root.addView(startButton);

        arCheck = new CheckBox(this);
        arCheck.setText("Use camera for position (ARCore VIO)\n- point the back camera forward, not at the floor/ceiling");
        arCheck.setChecked(prefs.getBoolean("ar", true));
        root.addView(arCheck);
        GLSurfaceView glView = new GLSurfaceView(this);
        root.addView(glView, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(24)));
        ar = new ArTracker(this, glView);

        statusView = new TextView(this);
        statusView.setTypeface(Typeface.MONOSPACE);
        statusView.setTextSize(12);
        statusView.setPadding(0, dp(12), 0, 0);
        root.addView(statusView);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(root);
        setContentView(scroll);
    }

    private TextView label(String s) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setPadding(0, dp(12), 0, 0);
        return t;
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    // ---------------------------------------------------------------- streaming

    private void startStreaming() {
        final String host = hostEdit.getText().toString().trim();
        final int port, rate;
        try {
            port = Integer.parseInt(portEdit.getText().toString().trim());
            rate = Math.max(1, Math.min(200, Integer.parseInt(rateEdit.getText().toString().trim())));
        } catch (NumberFormatException e) {
            lastError = "invalid port / rate";
            return;
        }
        getPreferences(MODE_PRIVATE).edit()
                .putString("host", host).putInt("port", port).putInt("rate", rate)
                .putBoolean("ar", arCheck.isChecked()).apply();

        registerSensors(rate);
        startLocation();
        if (arCheck.isChecked()) startAr();

        running = true;
        lastError = null;
        startButton.setText("STOP");
        sender = Executors.newSingleThreadScheduledExecutor();
        helloTarget = null;
        sender.execute(() -> {
            try {
                // Bind to the same port so a PC can "hello" us; replies to a PC-initiated flow
                // pass through stateful PC firewalls (e.g. ESET) that drop unsolicited UDP.
                DatagramSocket s = new DatagramSocket(null);
                s.setReuseAddress(true);
                try {
                    s.bind(new InetSocketAddress(port));
                } catch (Exception bindErr) {
                    s.close();
                    s = new DatagramSocket();
                    lastError = "port " + port + " busy; hello mode disabled";
                }
                s.setBroadcast(true);
                socket = s;
                startHelloListener(s);
                target = new InetSocketAddress(InetAddress.getByName(host), port);
            } catch (Exception e) {
                lastError = "socket: " + e.getMessage();
            }
        });
        sender.scheduleAtFixedRate(this::sendPacket, 100, Math.max(1, 1000L / rate), TimeUnit.MILLISECONDS);
    }

    private void startHelloListener(final DatagramSocket s) {
        helloThread = new Thread(() -> {
            byte[] buf = new byte[256];
            while (running && !s.isClosed()) {
                try {
                    DatagramPacket p = new DatagramPacket(buf, buf.length);
                    s.receive(p);
                    String msg = new String(p.getData(), 0, p.getLength(), StandardCharsets.UTF_8);
                    if (msg.startsWith(HELLO)) {
                        helloTarget = new InetSocketAddress(p.getAddress(), p.getPort());
                        lastHelloMs = System.currentTimeMillis();
                    }
                } catch (Exception e) {
                    if (s.isClosed()) return;
                }
            }
        }, "hello-listener");
        helloThread.setDaemon(true);
        helloThread.start();
    }

    private InetSocketAddress currentTarget() {
        InetSocketAddress h = helloTarget;
        if (h != null && System.currentTimeMillis() - lastHelloMs < HELLO_TIMEOUT_MS) return h;
        return target;
    }

    private static String localIps() {
        StringBuilder b = new StringBuilder();
        try {
            for (NetworkInterface ni : Collections.list(NetworkInterface.getNetworkInterfaces())) {
                if (!ni.isUp() || ni.isLoopback()) continue;
                for (InetAddress a : Collections.list(ni.getInetAddresses())) {
                    if (a instanceof Inet4Address) {
                        if (b.length() > 0) b.append(", ");
                        b.append(a.getHostAddress()).append(" (").append(ni.getName()).append(')');
                    }
                }
            }
        } catch (Exception ignored) { }
        return b.length() == 0 ? "-" : b.toString();
    }

    private void startAr() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
            return;
        }
        String err = ar.start();
        arActive = err == null;
        if (err != null) lastError = err;
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (running && arCheck.isChecked() && !arActive) startAr();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (arActive) {
            ar.stop();
            arActive = false;
        }
    }

    private void stopStreaming() {
        running = false;
        if (arActive) {
            ar.stop();
            arActive = false;
        }
        startButton.setText("START");
        if (sender != null) {
            sender.shutdownNow();
            sender = null;
        }
        DatagramSocket s = socket;
        socket = null;
        if (s != null) s.close();
        sensorManager.unregisterListener(this);
        locationManager.removeUpdates(this);
    }

    private void registerSensors(int rateHz) {
        int periodUs = Math.max(5000, 1000000 / Math.max(rateHz, 50));
        Sensor rv = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        quatSource = "rotation_vector";
        if (rv == null) {
            rv = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
            quatSource = "game_rotation_vector";
        }
        if (rv == null) quatSource = "none";
        else sensorManager.registerListener(this, rv, periodUs);
        int[] types = {Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE,
                Sensor.TYPE_MAGNETIC_FIELD, Sensor.TYPE_LINEAR_ACCELERATION, Sensor.TYPE_PRESSURE};
        for (int t : types) {
            Sensor s = sensorManager.getDefaultSensor(t);
            if (s != null) sensorManager.registerListener(this, s, periodUs);
        }
    }

    private void startLocation() {
        if (checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION}, REQ_LOCATION);
            return;
        }
        try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 100, 0, this, Looper.getMainLooper());
            }
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 1000, 0, this, Looper.getMainLooper());
            }
        } catch (SecurityException e) {
            lastError = "location: " + e.getMessage();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_LOCATION && running && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startLocation();
        }
        if (requestCode == REQ_CAMERA && running && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startAr();
        }
    }

    private void sendPacket() {
        DatagramSocket s = socket;
        InetSocketAddress dst = currentTarget();
        if (!running || s == null || dst == null) return;
        byte[] data = buildJson().getBytes(StandardCharsets.UTF_8);
        try {
            s.send(new DatagramPacket(data, data.length, dst));
        } catch (Exception e) {
            lastError = "send: " + e.getMessage();
        }
    }

    private synchronized String buildJson() {
        StringBuilder b = new StringBuilder(512);
        b.append("{\"v\":1,\"seq\":").append(seq++)
                .append(",\"t\":").append(System.currentTimeMillis());
        if (hasQuat) b.append(",\"q\":").append(arr(quat));
        if (hasAcc) b.append(",\"acc\":").append(arr(acc));
        if (hasGyr) b.append(",\"gyr\":").append(arr(gyr));
        if (hasMag) b.append(",\"mag\":").append(arr(mag));
        if (hasLin) b.append(",\"lin\":").append(arr(lin));
        if (!Float.isNaN(pressure)) b.append(",\"p\":").append(fmt(pressure));
        if (arActive) ar.appendJson(b);
        Location l = lastLocation;
        if (l != null) {
            b.append(",\"gps\":{\"lat\":").append(String.format(Locale.US, "%.8f", l.getLatitude()))
                    .append(",\"lon\":").append(String.format(Locale.US, "%.8f", l.getLongitude()))
                    .append(",\"alt\":").append(l.hasAltitude() ? fmt(l.getAltitude()) : "null")
                    .append(",\"acc\":").append(l.hasAccuracy() ? fmt(l.getAccuracy()) : "null")
                    .append(",\"spd\":").append(l.hasSpeed() ? fmt(l.getSpeed()) : "null")
                    .append(",\"brg\":").append(l.hasBearing() ? fmt(l.getBearing()) : "null")
                    .append(",\"prov\":\"").append(l.getProvider()).append('"')
                    .append(",\"t\":").append(l.getTime()).append('}');
        }
        b.append('}');
        return b.toString();
    }

    private static String fmt(double v) {
        return String.format(Locale.US, "%.5f", v);
    }

    private static String arr(float[] a) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) b.append(',');
            b.append(fmt(a[i]));
        }
        return b.append(']').toString();
    }

    // ---------------------------------------------------------------- callbacks

    @Override
    public synchronized void onSensorChanged(SensorEvent e) {
        switch (e.sensor.getType()) {
            case Sensor.TYPE_ROTATION_VECTOR:
            case Sensor.TYPE_GAME_ROTATION_VECTOR:
                // [w, x, y, z]: rotates device frame into world frame (X=East, Y=North, Z=Up)
                SensorManager.getQuaternionFromVector(quat, e.values);
                hasQuat = true;
                break;
            case Sensor.TYPE_ACCELEROMETER:
                System.arraycopy(e.values, 0, acc, 0, 3); hasAcc = true; break;
            case Sensor.TYPE_GYROSCOPE:
                System.arraycopy(e.values, 0, gyr, 0, 3); hasGyr = true; break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                System.arraycopy(e.values, 0, mag, 0, 3); hasMag = true; break;
            case Sensor.TYPE_LINEAR_ACCELERATION:
                System.arraycopy(e.values, 0, lin, 0, 3); hasLin = true; break;
            case Sensor.TYPE_PRESSURE:
                pressure = e.values[0]; break;
            default:
                break;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }

    @Override
    public void onLocationChanged(Location location) {
        Location prev = lastLocation;
        // Prefer GPS fixes; accept a network fix only if there has been no GPS fix for 5 s.
        if (prev != null && LocationManager.GPS_PROVIDER.equals(prev.getProvider())
                && !LocationManager.GPS_PROVIDER.equals(location.getProvider())
                && location.getTime() - prev.getTime() < 5000) {
            return;
        }
        lastLocation = location;
    }

    @Override public void onProviderEnabled(String provider) { }
    @Override public void onProviderDisabled(String provider) { }
    @Override public void onStatusChanged(String provider, int status, Bundle extras) { }

    // ---------------------------------------------------------------- status

    private final Runnable statusTick = new Runnable() {
        @Override
        public void run() {
            updateStatus();
            ui.postDelayed(this, 200);
        }
    };

    private synchronized void updateStatus() {
        StringBuilder b = new StringBuilder();
        b.append(running ? "STREAMING" : "stopped").append("   packets: ").append(seq).append('\n');
        if (lastError != null) b.append("ERROR: ").append(lastError).append('\n');
        b.append("this phone: ").append(localIps()).append('\n');
        if (running) {
            InetSocketAddress dst = currentTarget();
            boolean viaHello = dst != null && dst == helloTarget;
            b.append("sending to: ").append(dst == null ? "-" : dst.getAddress().getHostAddress() + ":" + dst.getPort())
                    .append(viaHello ? "  (PC hello)" : "").append('\n');
        }
        b.append("orientation src: ").append(quatSource).append('\n');
        if (ar != null) b.append(arActive ? ar.status() : "camera VIO: off").append('\n');
        b.append(String.format(Locale.US, "q   w%+.3f x%+.3f y%+.3f z%+.3f\n", quat[0], quat[1], quat[2], quat[3]));
        b.append(String.format(Locale.US, "acc %+7.2f %+7.2f %+7.2f m/s2\n", acc[0], acc[1], acc[2]));
        b.append(String.format(Locale.US, "gyr %+7.2f %+7.2f %+7.2f rad/s\n", gyr[0], gyr[1], gyr[2]));
        b.append(String.format(Locale.US, "mag %+7.1f %+7.1f %+7.1f uT\n", mag[0], mag[1], mag[2]));
        b.append(Float.isNaN(pressure) ? "pressure: n/a\n" : String.format(Locale.US, "pressure %.2f hPa\n", pressure));
        Location l = lastLocation;
        if (l == null) {
            b.append("GPS: waiting for fix...\n");
        } else {
            b.append(String.format(Locale.US, "GPS %.6f, %.6f\n alt %.1f m  acc %.1f m  [%s]\n",
                    l.getLatitude(), l.getLongitude(), l.getAltitude(), l.getAccuracy(), l.getProvider()));
        }
        statusView.setText(b.toString());
    }

    @Override
    protected void onDestroy() {
        stopStreaming();
        ar.close();
        ui.removeCallbacks(statusTick);
        super.onDestroy();
    }
}
