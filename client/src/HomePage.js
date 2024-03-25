import React, { useEffect, useRef, useState } from 'react';
import { Button, Image, ImageBackground, TouchableOpacity, Text, View } from 'react-native';
import EvilIcons from '@expo/vector-icons/EvilIcons';
import Button1 from './components/Button';
import { useNavigation } from '@react-navigation/native';
import axios from 'axios';

const PatternBg = { uri: 'https://e1.pxfuel.com/desktop-wallpaper/759/194/desktop-wallpaper-subtle-pride-phone-background-made-by-me-r-lgbt-thumbnail.jpg' };

function HomePage({ route }) {
    const navigation = useNavigation();
    const [images, setImages] = useState([]);
    const [dynamicIndex, setDynamicIndex] = useState(-1);

    const reorderImage = (flag) => {
        const len = images.length - 1;
        let newIndex = dynamicIndex + flag;
        if (newIndex < 0) newIndex = len;
        if (newIndex > len) newIndex = 0;
        setDynamicIndex(newIndex);
    };

    const setAttendance = async () => {
        var list = []
        images.map((image) => {
            image.label.filter(label => label !== 'UNKNOWN').map((label) => {
                list.push(label)
            })
        })

        console.log(list)

        try {
            const url = 'http://10.0.3.61:5000/store';

            // Create an object with the 'data' key and the list of data
            const requestData = { data: list };

            // Send POST request to the backend
            const response = await axios.post(url, requestData);

            // Log the response from the backend
            console.log(response.data);
        } catch (error) {
            // Handle errors
            console.error('Error:', error);
        }
    }

    const deleteImage = () => {
        const updatedImages = [...images];
        updatedImages.splice(dynamicIndex, 1);
        setImages(updatedImages);

        if (dynamicIndex >= updatedImages.length && dynamicIndex > 0) {
            setDynamicIndex(dynamicIndex - 1);
        }
    };

    useEffect(() => {
        if (route.params) {
            const { data, prediction } = route.params;
            setImages([...images, { uri: data, label: prediction.predicted_labels }]);
            setDynamicIndex(images.length);
        }
    }, [route.params]);

    return (
        <ImageBackground source={PatternBg} resizeMode='cover' style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <View style={{ width: '100%', alignItems: 'center' }}>
                <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginBottom: 10 }}>
                    <Button1
                        title={'Add an Image'}
                        icon='camera'
                        onPress={() => {
                            navigation.navigate('cameraPage');
                        }}
                    />
                </View>

                <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginBottom: 10 }}>
                    {images.length !== 0 && (
                        <EvilIcons
                            name='eye'
                            color='green'
                            style={{ fontSize: 20, marginRight: 5 }}
                        />
                    )}
                    <Text style={{ color: 'white' }}>No. of Images: {images.length}</Text>
                </View>

                <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginBottom: 10 }}>
                    {images.length !== 0 && (
                        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                            <EvilIcons
                                name='chevron-left'
                                color='#38a169'
                                size={55}
                                onPress={() => reorderImage(-1)}
                            />
                            <View style={{ position: 'relative', borderStyle: 'dashed', borderWidth: 2, borderColor: '#38a169', borderRadius: 8 }}>
                                <Button
                                    onPress={deleteImage}
                                    title="Remove"
                                    color="red"
                                    style={{ position: 'absolute', top: 0, right: 0, zIndex: 10 }}
                                />
                                <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
                                    {images.length !== 0 && <Image source={{ uri: images[dynamicIndex].uri }} style={{ width: 240, height: 240 }} />}
                                </View>
                            </View>

                            <EvilIcons
                                name='chevron-right'
                                color='#38a169'
                                size={55}
                                onPress={() => reorderImage(1)}
                            />
                        </View>
                    )}
                </View>

                <View style={{ marginBottom: 10 }}>
                    {images.length !== 0 && (
                        <View style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <Text style={{ color: 'gray', fontSize: 16 }}>
                                No. of Students:
                            </Text>
                            {images[dynamicIndex].label
                                .filter(label => label !== 'UNKNOWN')
                                .map((label, index) => (
                                    <Text key={index} style={{ color: '#38a169', fontSize: 16 }}>
                                        {label}
                                    </Text>
                                ))}
                        </View>
                    )}
                </View>


                <View style={{ marginBottom: 10 }}>
                    <Button
                        title='Mark Attendance'
                        onPress={() => {
                            setAttendance()
                            // Implement your logic for marking attendance here
                            console.log('Marking attendance...');
                        }}
                    />
                </View>
            </View>
        </ImageBackground>
    );
}

export default HomePage;
