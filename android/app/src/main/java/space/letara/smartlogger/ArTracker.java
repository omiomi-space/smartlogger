package space.letara.smartlogger;

import android.app.Activity;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.view.Surface;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;

import java.util.Locale;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Camera-based 6DoF tracking (visual-inertial odometry) with ARCore.
 * The camera image is never sent or stored; only the device pose is exported.
 *
 * Pose = Frame.getAndroidSensorPose(): the Android sensor frame (same axes as the IMU values)
 * expressed in ARCore's world frame (Y up / gravity aligned, arbitrary yaw, metres).
 */
public class ArTracker implements GLSurfaceView.Renderer {

    private final Activity activity;
    private final GLSurfaceView view;
    private Session session;
    private int textureId = -1;
    private boolean textureSet = false;
    private int width = 1, height = 1;
    private boolean geometryChanged = true;

    private volatile String state = "off";
    private volatile String failure = "";
    private long frames = 0;
    private long lastTs = 0;
    private double tsMs = 0;
    private final float[] pos = new float[3];
    private final float[] quat = {1, 0, 0, 0};   // w, x, y, z
    private boolean hasPose = false;

    public ArTracker(Activity activity, GLSurfaceView view) {
        this.activity = activity;
        this.view = view;
        view.setPreserveEGLContextOnPause(true);
        view.setEGLContextClientVersion(2);
        view.setRenderer(this);
        view.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    }

    /** Returns null on success, otherwise a message why AR is not available (yet). */
    public String start() {
        try {
            if (session == null) {
                ArCoreApk.Availability av = ArCoreApk.getInstance().checkAvailability(activity);
                if (av.isUnsupported()) {
                    state = "unsupported";
                    return "ARCore not supported on this device";
                }
                if (av != ArCoreApk.Availability.SUPPORTED_INSTALLED) {
                    ArCoreApk.InstallStatus st = ArCoreApk.getInstance().requestInstall(activity, true);
                    if (st == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                        state = "installing";
                        return "Install Google Play Services for AR, then press START again";
                    }
                }
                Session s = new Session(activity);
                Config cfg = new Config(s);
                cfg.setPlaneFindingMode(Config.PlaneFindingMode.DISABLED);
                cfg.setLightEstimationMode(Config.LightEstimationMode.DISABLED);
                cfg.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE);
                cfg.setFocusMode(Config.FocusMode.AUTO);
                s.configure(cfg);
                synchronized (this) {
                    session = s;
                    textureSet = false;
                }
            }
            session.resume();
            view.onResume();
            state = "starting";
            return null;
        } catch (Exception e) {
            state = "error";
            return "ARCore: " + e.getClass().getSimpleName() + " " + (e.getMessage() == null ? "" : e.getMessage());
        }
    }

    public void stop() {
        view.onPause();
        if (session != null) session.pause();
        state = "off";
    }

    public void close() {
        stop();
        if (session != null) {
            session.close();
            session = null;
        }
    }

    // ------------------------------------------------------------------ GL thread

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        int[] tex = new int[1];
        GLES20.glGenTextures(1, tex, 0);
        textureId = tex[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        textureSet = false;
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int w, int h) {
        width = w;
        height = h;
        geometryChanged = true;
        GLES20.glViewport(0, 0, w, h);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        Session s = session;
        boolean tracking = "TRACKING".equals(state);
        GLES20.glClearColor(0f, tracking ? 0.35f : 0.1f, tracking ? 0.15f : 0.1f, 1f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        if (s == null || textureId < 0) return;
        try {
            if (!textureSet) {
                s.setCameraTextureName(textureId);
                textureSet = true;
            }
            if (geometryChanged) {
                s.setDisplayGeometry(Surface.ROTATION_0, width, height);
                geometryChanged = false;
            }
            Frame frame = s.update();
            Camera cam = frame.getCamera();
            TrackingState ts = cam.getTrackingState();
            state = ts.name();
            long fts = frame.getTimestamp();
            if (ts == TrackingState.TRACKING && fts != lastTs) {
                lastTs = fts;
                Pose p = frame.getAndroidSensorPose();
                synchronized (this) {
                    pos[0] = p.tx(); pos[1] = p.ty(); pos[2] = p.tz();
                    quat[0] = p.qw(); quat[1] = p.qx(); quat[2] = p.qy(); quat[3] = p.qz();
                    frames++;
                    tsMs = fts / 1e6;
                    hasPose = true;
                }
                failure = "";
            } else {
                failure = cam.getTrackingFailureReason().name();
            }
        } catch (Exception e) {
            state = "error";
            failure = e.getClass().getSimpleName();
        }
    }

    // ------------------------------------------------------------------ export

    /** Appends ,"ar":{...} to the packet JSON (nothing if AR is off). */
    public synchronized void appendJson(StringBuilder b) {
        if (session == null) return;
        String st = state;
        b.append(",\"ar\":{\"st\":\"").append(st).append('"');
        if (!failure.isEmpty()) b.append(",\"why\":\"").append(failure).append('"');
        if (hasPose && "TRACKING".equals(st)) {
            b.append(",\"n\":").append(frames)
                    .append(String.format(Locale.US, ",\"ts\":%.3f", tsMs))
                    .append(String.format(Locale.US, ",\"p\":[%.4f,%.4f,%.4f]", pos[0], pos[1], pos[2]))
                    .append(String.format(Locale.US, ",\"q\":[%.5f,%.5f,%.5f,%.5f]", quat[0], quat[1], quat[2], quat[3]));
        }
        b.append('}');
    }

    public synchronized String status() {
        String s = "camera VIO: " + state;
        if (!failure.isEmpty()) s += " (" + failure + ")";
        if (hasPose) s += String.format(Locale.US, "\n  ar pos %+.2f %+.2f %+.2f m  frames %d", pos[0], pos[1], pos[2], frames);
        return s;
    }
}
