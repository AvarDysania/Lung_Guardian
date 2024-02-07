package com.example.liver_guardian;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.liver_guardian.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    Bitmap bitmap;
    TextView result, confidence;
    ImageView imageView;
    Button picture , select_button;
    int imageSize = 224;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        select_button=findViewById(R.id.button4);
        select_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,10);
            }
        });

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image)
    {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize *3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int [] intValues = new int[imageSize*imageSize];
            image.getPixels(intValues, 0 , image.getWidth() , 0 , 0 , image.getWidth(),image.getHeight());
            int pixel = 0;
            for(int i = 0 ;i<imageSize;i++)
            {
                for (int j = 0 ;j<imageSize;j++)
                {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val>>16) & 0xFF) *(1.f / 255.f));
                    byteBuffer.putFloat(((val>>8) & 0xFF) *(1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) *(1.f / 255.f));

                }
            }
            tensorBuffer.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(tensorBuffer);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0 ;i<confidences.length;i++)
            {
                if(confidences[i] > maxConfidence)
                {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"CANCER_BACTERIA","NORMAL"};
            result.setText(classes[maxPos]);
            

            String s = "";
            for(int i = 0 ;i< classes.length;i++)
            {
                s+= String.format("%s: %1f%%\n",classes[i], confidences[i] * 100);
            }
            confidence.setText(s);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if(requestCode==10)
        {
            if(data!=null)
            {
                Uri uri = data.getData();
                try{

                   bitmap=MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                   imageView.setImageBitmap(bitmap);

                   bitmap=Bitmap.createScaledBitmap(bitmap,imageSize,imageSize,false);
                   classifyImage(bitmap);

                } catch (IOException e)
                {
                   e.printStackTrace();
                }
            }
        }

        if (requestCode == 1 && resultCode == RESULT_OK) {
          Bitmap image = (Bitmap) data.getExtras().get("data");
          int dimension = Math.min(image.getWidth(),image.getHeight());
          image = ThumbnailUtils.extractThumbnail(image,dimension,dimension);
          imageView.setImageBitmap(image);

          image=Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
          classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}