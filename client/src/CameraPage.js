import React, { useEffect, useRef, useState } from 'react'
import { Button, ImageBackground, Text, TouchableOpacity, View } from 'react-native'
import { Entypo } from '@expo/vector-icons';
import * as MediaLibrary from 'expo-media-library';
import { useNavigation } from '@react-navigation/native';
import { Camera, CameraType } from 'expo-camera';
import axios from "axios"

function CameraPage() {
    const [hasCamerPermission, setHasCameraPermission] = useState(null);
    const [image, setImage] = useState(null);
    const [flash, setFlash] = useState(Camera.Constants.FlashMode.off);
    const navigation = useNavigation();
    const [type, setType] = useState(Camera.Constants.Type.back);
    const cameraRef = useRef(null);

    const takePicture = async () => {
        if (cameraRef) {
            try {
                const data = await cameraRef.current.takePictureAsync();

                const formData = new FormData();
                formData.append('image', {
                    uri: data.uri,
                    type: 'image/jpeg', // Adjust the type if needed
                    name: 'photo.jpg' // You can adjust the name of the file
                });

                // Send image to backend for processing
                const response = await axios.post('http://10.0.3.61:5000/predict', formData);
                const responseData = response.data;
                console.log(responseData)

                // Navigate to HomePage with image URI and response data
                navigation.navigate('homePage', { data: data.uri, prediction: responseData });
            } catch (error) {
                console.error('Error:', error);
            }
        }
    };


    useEffect(() => {
        (async () => {
            MediaLibrary.requestPermissionsAsync()
            const cameraStatus = await Camera.requestCameraPermissionsAsync();
            setHasCameraPermission(cameraStatus.status === 'granted')
        })();
    }, [])

    return (
        <ImageBackground>
            <View className='flex-none flex-col justify-center h-full w-full'>
                <View>
                    <Camera
                        className='h-full w-full'
                        flashMode={flash}
                        type={type}
                        ref={cameraRef}
                    ></Camera>

                    <View className='absolute bottom-0 w-full py-8 items-center'>
                        <TouchableOpacity
                            className='rounded-full bg-zinc-600 w-20 h-20 flex-none justify-center items-center'
                            onPress={takePicture}
                        >
                            <Entypo name='camera' size={45} />
                        </TouchableOpacity>
                    </View>
                </View>

            </View>
        </ImageBackground>
    )
}

export default CameraPage