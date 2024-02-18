import React, { useEffect, useRef, useState } from 'react'
import { Button, Image, ImageBackground, Text, View } from 'react-native'
import EvilIcons from '@expo/vector-icons/EvilIcons'
import { Camera, CameraType } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';
import Button1 from './components/Button';
import { useNavigation } from '@react-navigation/native';

const PatternBg = { uri: 'https://e1.pxfuel.com/desktop-wallpaper/759/194/desktop-wallpaper-subtle-pride-phone-background-made-by-me-r-lgbt-thumbnail.jpg' }


function HomePage({ route }) {
    const navigation = useNavigation();
    // const [hasCamerPermission, setHasCameraPermission]= useState(null);
    const [image, setImage] = useState([])
    const data = route.params;
    const [dynamicIndex, setDynamicIndedex] = useState(-1);

    const reorderImage = async (flag) => {
        const len = image.length - 1;
        if (flag == 1) {
            if (dynamicIndex === len)
                setDynamicIndedex(0);
            else
                setDynamicIndedex(dynamicIndex + 1);
            const temp = image[0];
            image.splice(0, 1);
            setImage([...image, temp]);
        }
        else {
            if (dynamicIndex <= 0)
                setDynamicIndedex(len);
            else
                await setDynamicIndedex(dynamicIndex - 1);
            const temp = image.pop();
            image.unshift(temp)
            setImage([...image]);
        }
    }

    const deletImage = async () => {
        const ind = image.length - 1;
        setDynamicIndedex(dynamicIndex - 1);
        image.splice(ind, 1);
        setImage([...image])
    }

    useEffect(() => {
        if (data) {
            setDynamicIndedex(dynamicIndex + 1);
            console.log(dynamicIndex);
            setImage([...image, data.data]);
        }
    }, [data])

    return (
        <ImageBackground source={PatternBg} resizeMode='cover' className='h-full w-full  flex-none justify-center items-center rotate-180'>
            <View className="w-full h-full flex-col-reverse justify-center items-center">
                <View style={{
                    borderStyle: 'dashed',
                    borderWidth: 2,
                    borderColor: '#38a169',
                    width: 240,
                    height: 240,
                    borderRadius: 8,
                    display: 'flex',
                    justifyContent: 'center',
                }}
                    className='rotate-180'
                >
                    <View>
                        <Button1 title={'Add an Image'} icon='camera'
                            onPress={() => {
                                navigation.navigate('cameraPage')
                            }}
                        />
                    </View>
                </View>

                <View className='h-10 flex-none flex-row items-center rotate-180'>
                    {image.length !== 0 ? (
                        image.map((imageUrl, index) => (
                            <View key={index}>
                                <EvilIcons
                                    name='eye'
                                    color={index === dynamicIndex ? 'green' : 'white'}
                                    style={{
                                        fontSize: 20
                                    }}
                                ></EvilIcons>
                            </View>
                        ))
                    ) : (
                        <View></View>
                    )}
                </View>

                <View className="flex-none flex-row justify-center items-center rotate-180">
                    <View><EvilIcons
                        name='chevron-left'
                        color='#38a169'
                        size={55}
                        onPress={() => { reorderImage(-1) }}
                    ></EvilIcons></View>
                    <View style={{
                        borderStyle: 'dashed',
                        borderWidth: 2,
                        borderColor: '#38a169',
                        width: 240,
                        height: 240,
                        borderRadius: 8,
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center'
                    }}>
                        <View className='absolute top-0 right-0'>
                            <EvilIcons
                                name='close-o'
                                style={{
                                    color: 'red',
                                    zIndex: 10,
                                    height: 30,
                                    fontSize: 35,
                                }}
                                onPress={() => {
                                    deletImage()
                                }}
                            ></EvilIcons>
                        </View>
                        <View className='absolute h-full w-full flex-none justify-center items-center'>
                            {image.length !== 0 ? (
                                image.map((image, index) => (
                                    <Image key={index} source={{ uri: image }} className='h-full w-full absolute' />
                                ))
                            ) : (
                                <Text className='text-white'>Image not available</Text>
                            )}
                        </View>
                    </View>
                    <View><EvilIcons
                        name='chevron-right'
                        style={{
                            color: '#38a169'
                        }}
                        size={55}
                        onPress={() => { reorderImage(1) }}
                    ></EvilIcons></View>
                </View>
                <View className='rotate-180 mb-3'>
                    <Text
                        className='text-gray-500 text-base font-medium text-lg'
                    >
                        No. of Students : <Text className='text-rose-400 font-black text-2xl'>{dynamicIndex}</Text>
                    </Text>
                </View>
                <View className='rotate-180 mb-8'>
                    <Button 
                        title={'Mark Attendance'}
                    >
                    </Button>
                </View>
            </View>
        </ImageBackground>
    )
}

export default HomePage