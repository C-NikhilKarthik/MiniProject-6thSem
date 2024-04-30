import React, { useState, useEffect } from 'react';
import { View, Text, Image, ImageBackground, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import EvilIcons from '@expo/vector-icons/EvilIcons';
import { MaterialIcons, Ionicons, FontAwesome, Feather } from '@expo/vector-icons';


const PatternBg = { uri: 'https://e1.pxfuel.com/desktop-wallpaper/759/194/desktop-wallpaper-subtle-pride-phone-background-made-by-me-r-lgbt-thumbnail.jpg' };

function HomePage() {
    const [images, setImages] = useState([]);
    const [dynamicIndex, setDynamicIndex] = useState(0);

    const reorderImage = (flag) => {
        const len = images.length - 1;
        let newIndex = dynamicIndex + flag;
        if (newIndex < 0) newIndex = len;
        if (newIndex > len) newIndex = 0;
        setDynamicIndex(newIndex);
    };


    useEffect(() => {
        (async () => {
            const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (status !== 'granted') {
                alert('Sorry, we need camera roll permissions to make this work!');
            }
        })();
    }, []);

    const pickImage = async () => {
        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Images,
                quality: 1,
            });

            if (!result.cancelled) {
                setImages([...images, result?.assets[0]?.uri]);
                uploadImage(result?.assets[0]?.uri)
            }
        } catch (error) {
            console.error('Error picking image:', error);
        }
    };

    const captureImage = async () => {
        try {
            const result = await ImagePicker.launchCameraAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Images,
                quality: 1,
            });

            if (!result.cancelled) {
                setImages([...images, result?.assets[0]?.uri]);
                uploadImage(result?.assets[0]?.uri)
            }
        } catch (error) {
            console.error('Error capturing image:', error);
        }
    };


    const uploadImage = async (image) => {
        try {
            const formData = new FormData();
            formData.append('image', {
                uri: image,
                type: 'image/jpeg',
                name: 'photo.png'
            });

            const response = await axios.post('http://10.0.3.61:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            console.log(response.data);
        } catch (error) {
            console.error('Error uploading image:', error);
        }
    };

    const storePredictions = async () => {
        try {
            const response = await axios.get('http://10.0.3.61:5000/save_predictions');
            console.log(response.data);
        } catch (error) {
            console.error('Error storing predictions:', error);
        }
    };

    return (
        <ImageBackground source={PatternBg} resizeMode='cover' style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <View style={{ width: '100%', alignItems: 'center' }}>
                <View style={{ flexDirection: 'column', justifyContent: 'center', alignItems: 'center', marginBottom: 10 }}>
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
                                    <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
                                        {images.length !== 0 && <Image source={{ uri: images[dynamicIndex] }} style={{ width: 240, height: 240 }} />}
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
                    <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginBottom: 10 }}>
                        <TouchableOpacity onPress={captureImage} style={{ flexDirection: 'row', alignItems: 'center' }}>
                            <EvilIcons name="camera" size={50} color="green" />
                            <Text style={{ marginLeft: 10, color: 'white', fontSize: 20 }}>Camera</Text>
                        </TouchableOpacity>
                    </View>

                    <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginBottom: 10 }}>
                        <TouchableOpacity onPress={pickImage} style={{ flexDirection: 'row', alignItems: 'center' }}>
                            <EvilIcons name="image" size={50} color="green" />
                            <Text style={{ marginLeft: 10, color: 'white', fontSize: 20 }}>Gallery</Text>
                        </TouchableOpacity>
                    </View>

                    <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginBottom: 10 }}>
                        <TouchableOpacity onPress={storePredictions} style={{ flexDirection: 'row', alignItems: 'center', backgroundColor: '#38a169', padding: 10, borderRadius: 8 }}>
                            <Text style={{ color: 'white', fontSize: 20 }}>Store Predictions</Text>
                        </TouchableOpacity>
                    </View>
                    {/* <Button title="Upload Images" onPress={uploadImage} /> */}
                </View>
            </View>
        </ImageBackground>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    image: {
        width: 300,
        height: 300,
        resizeMode: 'cover',
        marginBottom: 20,
    },
});

export default HomePage;
