import React, { useEffect, useRef, useState } from 'react';
import { View, TouchableOpacity, ImageBackground } from 'react-native';
import { Entypo } from '@expo/vector-icons';
import * as MediaLibrary from 'expo-media-library';
import { useNavigation } from '@react-navigation/native';
import { Camera } from 'expo-camera';
import axios from 'axios';
import { GestureHandlerRootView, PinchGestureHandler } from 'react-native-gesture-handler';
import { AutoFocus } from 'expo-camera/build/Camera.types';
import { State } from 'react-native-gesture-handler';

function CameraPage() {
    const [hasCameraPermission, setHasCameraPermission] = useState(null);
    const [flash, setFlash] = useState(Camera.Constants.FlashMode.off);
    const [type, setType] = useState(Camera.Constants.Type.back);
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [focusSquare, setFocusSquare] = useState({ visible: false, x: 0, y: 0 });
    const cameraRef = useRef(null);
    const navigation = useNavigation();

    useEffect(() => {
        (async () => {
            MediaLibrary.requestPermissionsAsync()
            const cameraStatus = await Camera.requestCameraPermissionsAsync();
            setHasCameraPermission(cameraStatus.status === 'granted')
        })();
    }, [])

    const takePicture = async () => {
        if (cameraRef) {
            try {
                const data = await cameraRef.current.takePictureAsync();

                const formData = new FormData();
                formData.append('image', {
                    uri: data.uri,
                    type: 'image/jpeg',
                    name: 'photo.jpg'
                });

                const response = await axios.post('http://10.0.3.61:5000/predict', formData);
                const responseData = response.data;

                navigation.navigate('homePage', { data: data.uri, prediction: responseData });
            } catch (error) {
                console.error('Error:', error);
            }
        }
    };

    // Function to handle touch events
    const handleTouch = (event) => {
        const { locationX, locationY } = event.nativeEvent;
        setFocusSquare({ visible: true, x: locationX, y: locationY });

        // Hide the square after 1 second
        setTimeout(() => {
            setFocusSquare((prevState) => ({ ...prevState, visible: false }));
        }, 1000);

        setIsRefreshing(true);
    };

    return (
        <GestureHandlerRootView style={{ flex: 1 }}>
            <PinchGestureHandler
                onHandlerStateChange={(event) => {
                    if (event.nativeEvent.state === State.END) {
                        const scale = event.nativeEvent.scale;
                        // Implement pinch to zoom functionality here
                    }
                }}>
                <View style={{ flex: 1 }}>
                    <Camera
                        style={{ flex: 1 }}
                        flashMode={flash}
                        type={type}
                        ref={cameraRef}
                        autoFocus={!isRefreshing ? AutoFocus.on : AutoFocus.off}
                        onTouchEnd={handleTouch} // Handle touch to set focus point
                    >
                        {focusSquare.visible && (
                            <View
                                style={[
                                    { position: 'absolute', width: 50, height: 50, borderWidth: 2, borderColor: 'white', backgroundColor: 'transparent' },
                                    { top: focusSquare.y - 25, left: focusSquare.x - 25 },
                                ]}
                            />
                        )}
                    </Camera>
                </View>
            </PinchGestureHandler>
            <View style={{ position: 'absolute', bottom: 20, width: '100%', justifyContent: 'center', alignItems: 'center' }}>
                <TouchableOpacity
                    style={{ backgroundColor: 'rgba(0,0,0,0.5)', padding: 10, borderRadius: 10 }}
                    onPress={takePicture}
                    disabled={isRefreshing}
                >
                    <Entypo name='camera' size={45} color='white' />
                </TouchableOpacity>
            </View>
        </GestureHandlerRootView>
    );
}

export default CameraPage;
