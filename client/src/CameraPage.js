import React, { useEffect, useRef, useState } from 'react'
import { Button, ImageBackground, Text, TouchableOpacity, View } from 'react-native'
import { Entypo } from '@expo/vector-icons';
import * as MediaLibrary from 'expo-media-library';
import { useNavigation } from '@react-navigation/native';
import { Camera, CameraType } from 'expo-camera';


// import Button from './components/Button'

function CameraPage() {
    const [hasCamerPermission, setHasCameraPermission]= useState(null);
    const [image, setImage] = useState(null);
    const [flash, setFlash] = useState(Camera.Constants.FlashMode.off);
    const navigation = useNavigation();
    const [type, setType] = useState(Camera.Constants.Type.back);
    const cameraRef = useRef(null);

    const takePicture = async ()=>{
        if(cameraRef){
            try{
                const data = await cameraRef.current.takePictureAsync();
                setImage(data.uri);
                navigation.navigate('homePage',{data:data.uri})
            }catch(e){
                console.log(e);
            }
        }
    }

    // if(hasCamerPermission===false)
    // {
    //     return <Text>No access to camera</Text>
    // }

    useEffect(()=>{
        (async()=>{
            MediaLibrary.requestPermissionsAsync()
            const cameraStatus = await Camera.requestCameraPermissionsAsync();
            setHasCameraPermission(cameraStatus.status==='granted')
        })();
    },[])

    return (
        <ImageBackground>
            <View className='flex-none flex-col jsutify-center border border-sky-500 h-full w-full'>
                <View>
                    <Camera 
                    className='h-5/6 w-full'
                    flashMode={flash}
                    type={type}
                    ref={cameraRef}
                    ></Camera>
                </View>
                <View className='flex-none items-center'>
                    <TouchableOpacity 
                    className='rounded-full bg-zinc-600 w-20 h-20 flex-none justify-center items-center'
                    onPress={takePicture}
                    >
                        <Entypo name='camera' size={45}/>
                    </TouchableOpacity>
                </View>
            </View>
        </ImageBackground>
    )
}

export default CameraPage